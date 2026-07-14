from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .brain_models import BrainEvent, CognitiveState, ConversationTurn, EventSource, WorkPriority


class EventJournal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
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
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO brain_events (
                    event_id, event_type, occurred_at, source, correlation_id,
                    causation_id, conversation_id, priority, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.event_type,
                    event.occurred_at.isoformat(),
                    event.source.value,
                    event.correlation_id,
                    event.causation_id,
                    event.conversation_id,
                    int(event.priority),
                    json.dumps(event.payload, separators=(",", ":"), default=str),
                ),
            )
            return event.model_copy(update={"sequence": int(cursor.lastrowid)})

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
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._row_to_event(row) for row in rows]

    def recent_turns(self, conversation_id: str = "default", limit: int = 20) -> list[ConversationTurn]:
        with self._lock, self._connect() as connection:
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
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS latest FROM brain_events WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["latest"])

    def recent_events(self, conversation_id: str = "default", limit: int = 100) -> list[BrainEvent]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM brain_events WHERE conversation_id = ?
                ORDER BY sequence DESC LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [self._row_to_event(row) for row in reversed(rows)]

    def restore_state(self, conversation_id: str = "default") -> CognitiveState:
        with self._lock, self._connect() as connection:
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
