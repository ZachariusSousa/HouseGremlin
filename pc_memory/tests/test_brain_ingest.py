"""Ingestion-driver tests: robot event stream (brain.db) -> pc_memory facts.

No LLM anywhere: conversion is deterministic, embeddings are hash-derived mocks,
and brain.db is a real (tiny) SQLite file in tmp_path so the read-only path is
exercised for real.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from pc_memory.app.brain_ingest import (
    STATE_KEY,
    convert_event,
    convert_row,
    ingest_brain,
    load_state,
)
from pc_memory.app.db import connect, init_schema


class MockEmbed:
    """Deterministic embedding stand-in (hash-derived vectors)."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed(self, texts: list[str], **kwargs):
        if not texts:
            return []
        out = []
        for text in texts:
            digest = hashlib.sha256(str(text).encode("utf-8")).digest()
            out.append([b / 255.0 for b in digest[: self.dim]])
        return out


def _make_brain_db(path: Path, events: list[dict]) -> None:
    """Create a brain.db with the real pc_brain schema and the given events."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE brain_events (
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
        )"""
    )
    for i, ev in enumerate(events, start=1):
        conn.execute(
            "INSERT INTO brain_events (sequence, event_id, event_type, occurred_at, source,"
            " correlation_id, causation_id, conversation_id, priority, payload_json)"
            " VALUES (?, ?, ?, ?, ?, ?, NULL, 'default', 20, ?)",
            (i, ev["event_id"], ev["event_type"], ev["occurred_at"], ev.get("source", "system"),
             "corr-1", json.dumps(ev["payload"])),
        )
    conn.commit()
    conn.close()


EVENTS = [
    {
        "event_id": "ev-fault-1",
        "event_type": "fault.registered",
        "occurred_at": "2026-09-15T14:16:51.311798+00:00",
        "payload": {
            "fault": {
                "source": "esp_heartbeat",
                "severity": "critical",
                "message": "control channel not ready",
                "timestamp": "2026-09-15T14:16:51.309798Z",
            }
        },
    },
    {
        "event_id": "ev-action-1",
        "event_type": "action.failed",
        "occurred_at": "2026-09-15T14:36:17.991266+00:00",
        "source": "firmware",
        "payload": {
            "action": {"movement": {"direction": "stop", "speed": 180}},
            "error": "TimeoutError",
        },
    },
    {
        "event_id": "ev-state-1",
        "event_type": "state.changed",
        "occurred_at": "2026-09-15T14:36:18.000000+00:00",
        "payload": {
            "state": {
                "conversation": "idle",
                "body": "fault",
                "active_goal": None,
                "connectivity": "online",
                "safety": "fault",
            }
        },
    },
]


@pytest.fixture()
def env(tmp_path):
    conn = connect(tmp_path / "mem.db")
    init_schema(conn)
    brain_db = tmp_path / "brain.db"
    _make_brain_db(brain_db, EVENTS)
    state_file = tmp_path / "state.json"
    yield SimpleNamespace(
        conn=conn, brain_db=brain_db, state_file=state_file, embed=MockEmbed()
    )
    conn.close()


def test_convert_event_known_shapes():
    fault = convert_event(
        "fault.registered",
        {"fault": {"source": "esp_heartbeat", "severity": "critical",
                   "message": "control channel not ready"}},
    )
    assert fault == (
        "A critical fault was registered from source 'esp_heartbeat': "
        "control channel not ready"
    )

    action = convert_event(
        "action.failed",
        {"action": {"movement": {"direction": "stop", "speed": 180}},
         "error": "TimeoutError"},
    )
    assert action == (
        "The robot failed an movement 'stop' (speed 180) action with error 'TimeoutError'"
    )

    state = convert_event(
        "state.changed",
        {"state": {"conversation": "idle", "body": "fault", "safety": "fault",
                   "connectivity": "online"}},
    )
    assert state == (
        "The robot's state is conversation=idle, body=fault, safety=fault, "
        "connectivity=online"
    )

    eyes = convert_event("eyes.heartbeat.failed", {"error": "control channel not ready"})
    assert eyes == "Eyes heartbeat failed: control channel not ready"


def test_convert_event_unknown_shape_returns_none():
    assert convert_event("mystery.event", {"whatever": 1}) is None
    assert convert_event("fault.registered", None) is None
    assert convert_event("state.changed", {}) is None


def test_convert_row_carries_metadata(tmp_path):
    # Build a real sqlite3.Row (the same type read_events returns).
    conn = sqlite3.connect(str(tmp_path / "row.db"))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE t (sequence, event_id, event_type, occurred_at, source, payload_json)"
    )
    conn.execute(
        "INSERT INTO t VALUES (?, ?, ?, ?, ?, ?)",
        (7, "ev-x", "eyes.heartbeat.failed", "2026-09-15T00:00:00+00:00",
         "firmware", json.dumps({"error": "boom"})),
    )
    row = conn.execute("SELECT * FROM t").fetchone()
    conn.close()

    claim, meta = convert_row(row)
    assert claim == "Eyes heartbeat failed: boom"
    assert meta["event_id"] == "ev-x"
    assert meta["sequence"] == 7


def test_ingest_brain_writes_facts_with_provenance(env):
    summary = ingest_brain(env.brain_db, env.state_file, env.conn, embed=env.embed)
    assert summary["events_seen"] == 3
    assert summary["facts_added"] == 3
    assert summary["facts_merged"] == 0
    assert summary["skipped_unknown"] == 0

    node_count = env.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert node_count == 3
    # Every fact carries provenance pointing back at the source brain event.
    refs = [r[0] for r in env.conn.execute(
        "SELECT source_ref FROM provenance ORDER BY id")]
    assert refs == [
        "brain_event:ev-fault-1",
        "brain_event:ev-action-1",
        "brain_event:ev-state-1",
    ]
    kinds = {r[0] for r in env.conn.execute(
        "SELECT source_kind FROM provenance")}
    assert kinds == {"manual"}
    # Embeddings were stored (hybrid mode).
    embedded = env.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL").fetchone()[0]
    assert embedded == 3


def test_ingest_brain_idempotent_across_state_reset(env):
    first = ingest_brain(env.brain_db, env.state_file, env.conn, embed=env.embed)
    assert first["facts_added"] == 3

    # Second run: state advanced -> nothing new.
    second = ingest_brain(env.brain_db, env.state_file, env.conn, embed=env.embed)
    assert second["events_seen"] == 0
    assert second["facts_added"] == 0

    # Simulate a crash that lost the watermark: reset state to force a full
    # re-read. Per-event idempotency must skip every already-recorded event so
    # occurrences are never double-counted, even though the claims would match.
    env.state_file.write_text(json.dumps({STATE_KEY: 0}))
    third = ingest_brain(env.brain_db, env.state_file, env.conn, embed=env.embed)
    assert third["events_seen"] == 3
    assert third["facts_added"] == 0
    assert third["facts_merged"] == 0
    assert third["skipped_already_ingested"] == 3
    node_count = env.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert node_count == 3  # no duplicate nodes
    prov_count = env.conn.execute("SELECT COUNT(*) FROM provenance").fetchone()[0]
    assert prov_count == 3  # each fact still has exactly ONE provenance row


def test_ingest_brain_merges_genuine_repeats(env):
    """A NEW event (fresh id) with an IDENTICAL claim merges into the existing
    node and accumulates a second provenance row — occurrence counting works."""
    first = ingest_brain(env.brain_db, env.state_file, env.conn, embed=env.embed)
    assert first["facts_added"] == 3

    # Append a brand-new fault event with the SAME payload as ev-fault-1 but a
    # fresh event_id -> identical claim must merge into the existing node.
    conn = sqlite3.connect(str(env.brain_db))
    conn.execute(
        "INSERT INTO brain_events (sequence, event_id, event_type, occurred_at, source,"
        " correlation_id, causation_id, conversation_id, priority, payload_json)"
        " VALUES (?, ?, ?, ?, ?, ?, NULL, 'default', 20, ?)",
        (100, "ev-fault-1-repeat", "fault.registered", "2026-09-15T15:00:00+00:00",
         "system", "corr-1", json.dumps(EVENTS[0]["payload"])),
    )
    conn.commit()
    conn.close()

    second = ingest_brain(env.brain_db, env.state_file, env.conn, embed=env.embed)
    assert second["events_seen"] == 1
    assert second["facts_added"] == 0      # no new node...
    assert second["facts_merged"] == 1     # ...merged into the existing fault fact
    node_count = env.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert node_count == 3                 # still 3 nodes
    occ = env.conn.execute(
        "SELECT COUNT(*) FROM provenance WHERE source_ref='brain_event:ev-fault-1-repeat'"
    ).fetchone()[0]
    assert occ == 1                         # the repeat recorded exactly once


def test_ingest_brain_dry_run_writes_nothing(env):
    summary = ingest_brain(
        env.brain_db, env.state_file, env.conn, embed=env.embed, dry_run=True
    )
    assert summary["dry_run"] is True
    assert summary["events_seen"] == 3
    assert "claims" in summary and len(summary["claims"]) == 3
    assert env.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 0
    # Dry run must not advance state (a real run afterwards still sees all events).
    assert load_state(env.state_file) == {}


def test_ingest_brain_missing_db_raises(tmp_path):
    conn = connect(tmp_path / "mem.db")
    init_schema(conn)
    try:
        with pytest.raises(FileNotFoundError):
            ingest_brain(tmp_path / "nope.db", tmp_path / "state.json", conn)
    finally:
        conn.close()


def test_ingest_brain_skips_unknown_event_types(env, tmp_path):
    # Append an unknown event type to the brain db.
    conn = sqlite3.connect(str(env.brain_db))
    conn.execute(
        "INSERT INTO brain_events (sequence, event_id, event_type, occurred_at, source,"
        " correlation_id, causation_id, conversation_id, priority, payload_json)"
        " VALUES (?, ?, ?, ?, ?, ?, NULL, 'default', 20, ?)",
        (99, "ev-unknown", "mystery.event", "2026-09-15T00:00:00+00:00",
         "system", "corr-1", json.dumps({"x": 1})),
    )
    conn.commit()
    conn.close()

    summary = ingest_brain(env.brain_db, env.state_file, env.conn, embed=env.embed)
    assert summary["events_seen"] == 4
    assert summary["skipped_unknown"] == 1
    assert summary["facts_added"] == 3


def test_state_roundtrip(tmp_path):
    from pc_memory.app.brain_ingest import save_state

    state_file = tmp_path / "state.json"
    assert load_state(state_file) == {}
    save_state(state_file, {STATE_KEY: 42})
    assert load_state(state_file) == {STATE_KEY: 42}
