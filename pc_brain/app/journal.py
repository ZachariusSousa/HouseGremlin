from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .brain_models import BrainEvent, CognitiveState, ConversationTurn, EventSource, WorkPriority


@dataclass
class _WriteRequest:
    event: BrainEvent
    completed: threading.Event | None
    result: BrainEvent | None = None
    error: Exception | None = None


class EventJournal:
    def __init__(self, path: Path, queue_limit: int = 1000):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._read_lock = threading.RLock()
        self._queue: queue.Queue[_WriteRequest | None] = queue.Queue(
            maxsize=max(10, queue_limit)
        )
        self._drops = 0
        self._written = 0
        self._stopping = False
        self._initialize()
        self._writer = threading.Thread(
            target=self._writer_loop,
            name="robit-journal-writer",
            daemon=True,
        )
        self._writer.start()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._read_lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS brain_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    correlation_id TEXT NOT NULL,
                    causation_id TEXT,
                    conversation_id TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_brain_events_conversation_sequence
                    ON brain_events(conversation_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_brain_events_correlation
                    ON brain_events(correlation_id, sequence);
                """
            )

    def append(self, event: BrainEvent) -> BrainEvent:
        droppable = event.event_type in {
            "tracking.sample",
            "telemetry.sample",
            "event_loop.sample",
        }
        completed = None if droppable else threading.Event()
        request = _WriteRequest(event=event, completed=completed)
        try:
            self._queue.put(request, block=not droppable, timeout=5.0 if not droppable else None)
        except queue.Full:
            self._drops += 1
            return event
        if completed is None:
            return event
        if not completed.wait(timeout=5.0):
            raise TimeoutError("journal commit timed out")
        if request.error is not None:
            raise request.error
        return request.result or event

    def _writer_loop(self) -> None:
        connection = self._connect()
        try:
            while True:
                first = self._queue.get()
                if first is None:
                    self._queue.task_done()
                    break
                batch = [first]
                # Flush promptly for interactive events while still collecting
                # bursts; the hard upper bound remains well below 100 ms.
                deadline = time.monotonic() + 0.01
                while len(batch) < 50:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = self._queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if item is None:
                        self._queue.task_done()
                        self._stopping = True
                        break
                    batch.append(item)
                try:
                    connection.execute("BEGIN")
                    for request in batch:
                        cursor = connection.execute(
                            """
                            INSERT INTO brain_events (
                                event_id, event_type, occurred_at, source, correlation_id,
                                causation_id, conversation_id, priority, payload_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                request.event.event_id,
                                request.event.event_type,
                                request.event.occurred_at.isoformat(),
                                request.event.source.value,
                                request.event.correlation_id,
                                request.event.causation_id,
                                request.event.conversation_id,
                                int(request.event.priority),
                                json.dumps(
                                    request.event.payload,
                                    separators=(",", ":"),
                                    default=str,
                                ),
                            ),
                        )
                        request.result = request.event.model_copy(
                            update={"sequence": int(cursor.lastrowid)}
                        )
                    connection.commit()
                    self._written += len(batch)
                except Exception as exc:
                    connection.rollback()
                    for request in batch:
                        request.error = exc
                finally:
                    for request in batch:
                        if request.completed is not None:
                            request.completed.set()
                        self._queue.task_done()
                if self._stopping and self._queue.empty():
                    break
        finally:
            connection.close()

    def close(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self._queue.put(None)
        self._writer.join(timeout=5.0)

    def status(self) -> dict[str, int]:
        return {
            "depth": self._queue.qsize(),
            "capacity": self._queue.maxsize,
            "drops": self._drops,
            "written": self._written,
        }

    def list_events(
        self,
        conversation_id: str = "default",
        after_sequence: int = 0,
        limit: int = 100,
        correlation_id: str | None = None,
    ) -> list[BrainEvent]:
        query = "SELECT * FROM brain_events WHERE conversation_id = ? AND sequence > ?"
        values: list[object] = [conversation_id, after_sequence]
        if correlation_id:
            query += " AND correlation_id = ?"
            values.append(correlation_id)
        query += " ORDER BY sequence ASC LIMIT ?"
        values.append(limit)
        with self._read_lock, self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._row_to_event(row) for row in rows]

    def recent_turns(self, conversation_id: str = "default", limit: int = 20) -> list[ConversationTurn]:
        with self._read_lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM brain_events
                WHERE conversation_id = ?
                  AND event_type IN ('conversation.user.completed', 'conversation.assistant.completed')
                ORDER BY sequence DESC LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        turns = []
        for row in reversed(rows):
            payload = json.loads(row["payload_json"])
            text = str(payload.get("text") or "").strip()
            if text:
                turns.append(
                    ConversationTurn(
                        role="user" if row["event_type"] == "conversation.user.completed" else "assistant",
                        text=text,
                        correlation_id=row["correlation_id"],
                        sequence=row["sequence"],
                    )
                )
        return turns

    def latest_sequence(self, conversation_id: str = "default") -> int:
        with self._read_lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS latest FROM brain_events WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["latest"])

    def recent_events(self, conversation_id: str = "default", limit: int = 100) -> list[BrainEvent]:
        with self._read_lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM brain_events WHERE conversation_id = ?
                ORDER BY sequence DESC LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [self._row_to_event(row) for row in reversed(rows)]

    def restore_state(self, conversation_id: str = "default") -> CognitiveState:
        with self._read_lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM brain_events
                WHERE conversation_id = ? AND event_type = 'state.changed'
                ORDER BY sequence DESC LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        if row:
            payload = json.loads(row["payload_json"])
            try:
                return CognitiveState.model_validate(payload.get("state", payload))
            except ValueError:
                pass
        return CognitiveState(conversation_id=conversation_id)

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> BrainEvent:
        return BrainEvent(
            sequence=row["sequence"],
            event_id=row["event_id"],
            event_type=row["event_type"],
            occurred_at=row["occurred_at"],
            source=EventSource(row["source"]),
            correlation_id=row["correlation_id"],
            causation_id=row["causation_id"],
            conversation_id=row["conversation_id"],
            priority=WorkPriority(row["priority"]),
            payload=json.loads(row["payload_json"]),
        )
