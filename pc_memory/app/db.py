"""SQLite connection helper and schema init for pc_memory.

Schema: nodes (entity/fact), typed edges, provenance, retrieval_traces, url_cache,
plus an FTS5 external-content table over nodes.text kept in sync by triggers.
"""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('entity', 'fact')),
    text TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    embedding BLOB,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    dst INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    UNIQUE (src, dst, type)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);

CREATE TABLE IF NOT EXISTS provenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('web', 'owner_qa', 'manual')),
    source_ref TEXT,
    snippet TEXT,
    confidence REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_provenance_node ON provenance(node_id);

CREATE TABLE IF NOT EXISTS retrieval_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    seed_ids TEXT NOT NULL DEFAULT '[]',
    decisions TEXT NOT NULL DEFAULT '[]',
    visited_ids TEXT NOT NULL DEFAULT '[]',
    context TEXT NOT NULL DEFAULT '',
    feedback TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS url_cache (
    url TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    text,
    content='nodes',
    content_rowid='id'
);

-- Keep the external-content FTS index in sync with nodes.text.
CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO nodes_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection, creating parent dirs as needed."""
    path = Path(db_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: FastAPI runs sync endpoints on a thread-pool
    # worker while the connection is created at startup; access is serialized
    # by Service.lock (see main.py).
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables, indexes, the FTS5 table and sync triggers (idempotent)."""
    conn.executescript(SCHEMA)
    conn.commit()


def rebuild_fts(conn: sqlite3.Connection) -> None:
    """Re-sync the FTS5 index (and anything derived from source rows) from nodes."""
    conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES ('rebuild')")
    conn.commit()
