"""Background ingestion driver: robot event stream (brain.db) -> pc_memory facts.

The robot's own telemetry is the fuel for the memory graph. `brain_events` in
the pc_brain SQLite DB holds structured events (faults, action lifecycles, state
transitions); this module converts each one to a stable claim sentence and
stores it as a canonical fact node with provenance pointing back at the source
event (`source_kind='manual'`, `source_ref='brain_event:<event_id>'`).

No LLM is involved: conversion is deterministic (the events are already
structured), so the driver runs while the chat LLM is down and never competes
for VRAM. Embeddings come from the dedicated embedding endpoint when available
and degrade to FTS5-only otherwise, exactly like the rest of the service.

Repeated occurrences of the same event type with identical payloads produce an
identical claim sentence, so the store merges them into one fact node and
accumulates provenance rows — "happened N times" is queryable via provenance
count without any dedup judge.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

from pc_memory.app.store import add_fact

logger = logging.getLogger("pc_memory.brain_ingest")

DEFAULT_BRAIN_DB = "pc_brain/data/brain.db"
# State file lives next to the memory DB (resolved in the CLI), not CWD-relative,
# so ingestion progress is independent of where the command is run from.
DEFAULT_STATE_FILE = "ingest_state.json"
STATE_KEY = "last_sequence"


# ---------------------------------------------------------------------------
# deterministic event -> claim conversion (no LLM)
# ---------------------------------------------------------------------------


def _action_text(action: Any) -> str | None:
    if not isinstance(action, dict):
        return None
    movement = action.get("movement")
    if isinstance(movement, dict):
        direction = str(movement.get("direction", "")).strip()
        speed = movement.get("speed")
        if direction:
            return f"movement '{direction}' (speed {speed})" if speed is not None else f"movement '{direction}'"
    for key in sorted(action):
        if isinstance(action[key], dict):
            nested = _action_text(action[key])
            if nested:
                return nested
    return None


def convert_event(event_type: str, payload: Any) -> str | None:
    """Convert one structured brain event into a stable claim sentence.

    Returns None for unknown/empty shapes (the caller skips those). The claim
    deliberately omits timestamps so that repeated occurrences of the same
    event merge into a single fact node; the per-occurrence time is preserved
    in the provenance row's `occurred_at` field instead.
    """
    if not isinstance(payload, dict):
        return None

    if event_type == "fault.registered":
        fault = payload.get("fault")
        if isinstance(fault, dict):
            source = str(fault.get("source", "")).strip() or "unknown"
            severity = str(fault.get("severity", "")).strip() or "unspecified"
            message = str(fault.get("message", "")).strip() or "no message"
            return f"A {severity} fault was registered from source '{source}': {message}"

    if event_type == "eyes.heartbeat.failed":
        error = str(payload.get("error", "")).strip() or "unknown error"
        return f"Eyes heartbeat failed: {error}"

    if event_type == "eyes.expression.changed":
        expression = str(payload.get("expression", "")).strip()
        if not expression:
            eyes = payload.get("eyes")
            expression = str(eyes.get("effective_expression", "")).strip() if isinstance(eyes, dict) else ""
        if expression:
            return f"The robot's effective eye expression is now '{expression}'"

    if event_type in ("action.proposed", "action.approved", "action.failed"):
        action_text = _action_text(payload.get("action"))
        if not action_text:
            return None
        verb = {"action.proposed": "proposed", "action.approved": "approved", "action.failed": "failed"}[event_type]
        claim = f"The robot {verb} an {action_text} action"
        error = str(payload.get("error", "")).strip()
        if event_type == "action.failed" and error:
            claim += f" with error '{error}'"
        return claim

    if event_type == "state.changed":
        state = payload.get("state")
        if not isinstance(state, dict):
            return None
        conversation = str(state.get("conversation", "")).strip() or "unknown"
        body = str(state.get("body", "")).strip() or "unknown"
        safety = str(state.get("safety", "")).strip() or "unknown"
        connectivity = str(state.get("connectivity", "")).strip() or "unknown"
        return (
            f"The robot's state is conversation={conversation}, body={body}, "
            f"safety={safety}, connectivity={connectivity}"
        )

    return None


# ---------------------------------------------------------------------------
# brain.db access (read-only) + conversion pass
# ---------------------------------------------------------------------------


def open_brain_db(path: str | Path) -> sqlite3.Connection:
    """Open the robot's brain.db read-only; raises if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"brain db not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def read_events(conn: sqlite3.Connection, after_sequence: int = 0) -> list[sqlite3.Row]:
    """All brain_events with sequence > after_sequence, in sequence order."""
    return conn.execute(
        "SELECT sequence, event_id, event_type, occurred_at, source, payload_json "
        "FROM brain_events WHERE sequence > ? ORDER BY sequence",
        (int(after_sequence),),
    ).fetchall()


def already_ingested(store_conn: sqlite3.Connection, event_id: str) -> bool:
    """True if this brain event's provenance row is already in the store.

    Makes ingestion idempotent per-event independent of the state file: a crash +
    restart (or a manual state reset) re-reads the same events but never records
    their occurrences twice, so repeated telemetry can't inflate occurrence counts
    or duplicate provenance rows even if the sequence watermark is lost.
    """
    row = store_conn.execute(
        "SELECT 1 FROM provenance WHERE source_ref = ? LIMIT 1",
        (f"brain_event:{event_id}",),
    ).fetchone()
    return row is not None


def convert_row(row: sqlite3.Row) -> tuple[str | None, dict]:
    """One brain_event row -> (claim | None, metadata). Metadata always carries
    event_id/type/occurred_at/source so provenance is complete even when the
    claim itself is empty (unknown shape -> skipped by callers)."""
    try:
        payload = json.loads(row["payload_json"])
    except (json.JSONDecodeError, TypeError):
        payload = None
    claim = convert_event(row["event_type"], payload)
    meta = {
        "event_id": row["event_id"],
        "event_type": row["event_type"],
        "occurred_at": row["occurred_at"],
        "source": row["source"],
        "sequence": row["sequence"],
    }
    return claim, meta


# ---------------------------------------------------------------------------
# state (last ingested sequence) — small JSON file, human-inspectable
# ---------------------------------------------------------------------------


def load_state(state_path: str | Path) -> dict[str, Any]:
    path = Path(state_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("state file unreadable (%s); starting fresh", path)
        return {}
    return data if isinstance(data, dict) else {}


def save_state(state_path: str | Path, state: dict[str, Any]) -> None:
    path = Path(state_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# ingest pass + watch loop
# ---------------------------------------------------------------------------


def ingest_brain(
    brain_db: str | Path,
    state_path: str | Path,
    store_conn: sqlite3.Connection,
    *,
    embed: Any = None,
    confidence: float = 0.7,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ingest new brain events into the pc_memory store.

    Reads only events with sequence > state['last_sequence'], converts each to
    a claim, and stores it via `add_fact` (exact-match dedup merges repeated
    occurrences into one node, accumulating provenance rows). Returns a summary;
    never raises for per-event conversion problems (those are skipped and
    counted), only for missing brain.db / state write failures.
    """
    state = load_state(state_path)
    last_seq = int(state.get(STATE_KEY, 0))
    brain = open_brain_db(brain_db)
    try:
        rows = read_events(brain, after_sequence=last_seq)
    finally:
        brain.close()

    added = 0
    merged = 0
    skipped = 0
    skipped_already = 0
    results: list[dict[str, Any]] = []
    for row in rows:
        claim, meta = convert_row(row)
        if not claim:
            skipped += 1
            continue
        source_ref = f"brain_event:{meta['event_id']}"
        provenance = {
            "source_kind": "manual",
            "source_ref": source_ref,
            "snippet": f"{meta['event_type']} at {meta['occurred_at']}: {claim[:400]}",
        }
        if dry_run:
            results.append({"sequence": meta["sequence"], "event_type": meta["event_type"], "claim": claim})
            continue
        # Per-event idempotency: the sequence watermark is the primary gate, but a
        # crash/reset can lose it. Never record an event's occurrence twice.
        if already_ingested(store_conn, meta["event_id"]):
            skipped_already += 1
            continue
        embedding = None
        if embed is not None:
            vecs = embed.embed([claim])
            if vecs and len(vecs) == 1:
                embedding = vecs[0]
        result = add_fact(
            store_conn,
            claim,
            confidence=confidence,
            provenance=provenance,
            embedding=embedding,
        )
        if result.verdict == "duplicate":
            merged += 1
        else:
            added += 1
        results.append(
            {
                "sequence": meta["sequence"],
                "event_type": meta["event_type"],
                "claim": claim,
                "node_id": result.node_id,
                "verdict": result.verdict,
            }
        )

    if rows and not dry_run:
        state[STATE_KEY] = max(int(r["sequence"]) for r in rows)
        save_state(state_path, state)

    summary = {
        "dry_run": dry_run,
        "events_seen": len(rows),
        "facts_added": added,
        "facts_merged": merged,
        "skipped_unknown": skipped,
        "skipped_already_ingested": skipped_already,
        "last_sequence": int(state.get(STATE_KEY, 0)),
    }
    if dry_run:
        summary["claims"] = results
    return summary


def watch_brain(
    brain_db: str | Path,
    state_path: str | Path,
    store_conn: sqlite3.Connection,
    *,
    embed: Any = None,
    confidence: float = 0.7,
    poll_interval: float = 15.0,
    stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Poll brain.db for new events forever (until `stop()` is true).

    Each cycle ingests everything newer than the saved state; a missing or
    locked brain.db is logged and retried on the next cycle rather than
    crashing the loop. Returns the summary of the last completed cycle.
    """
    sleep = max(1.0, float(poll_interval))
    while stop is None or not stop():
        try:
            summary = ingest_brain(
                brain_db, state_path, store_conn, embed=embed, confidence=confidence
            )
            if summary["events_seen"]:
                logger.info("ingested %s", summary)
        except FileNotFoundError as exc:
            logger.warning("brain db unavailable (%s); retrying next cycle", exc)
        except sqlite3.Error as exc:
            logger.warning("brain db read failed (%s); retrying next cycle", exc)
        time.sleep(sleep)
    return summary  # type: ignore[possibly-undefined]
