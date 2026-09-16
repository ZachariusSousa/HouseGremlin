"""Phase 0 smoke tests: package imports, settings, schema + FTS5 sync in a temp DB."""

from pc_memory.app.config import Settings, load_settings
from pc_memory.app.db import connect, init_schema, rebuild_fts


def _tables(conn) -> set:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')"
        )
    }


def _fts_ids(conn, term: str) -> list:
    return [
        row[0]
        for row in conn.execute(
            "SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH ? ORDER BY rowid", (term,)
        )
    ]


def test_settings_defaults(monkeypatch):
    for name in (
        "MEMORY_PORT",
        "MEMORY_DB_PATH",
        "MEMORY_LLM_BASE_URL",
        "MEMORY_LLM_MODEL",
        "MEMORY_LLM_TIMEOUT",
        "MEMORY_CONTEXT_BUDGET_CHARS",
        "MEMORY_MAX_HOPS",
        "MEMORY_BEAM",
    ):
        monkeypatch.delenv(name, raising=False)
    s = load_settings()
    assert isinstance(s, Settings)
    assert s.port == 8092
    assert s.db_path.parts[-2:] == ("data", "memory.db")
    assert s.llm_base_url == "http://127.0.0.1:8081/v1"
    assert s.context_budget_chars == 6000
    assert s.max_hops == 3
    assert s.beam == 6


def test_settings_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_PORT", "9999")
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "custom.db"))
    monkeypatch.setenv("MEMORY_CONTEXT_BUDGET_CHARS", "1234")
    s = load_settings()
    assert s.port == 9999
    assert s.db_path == tmp_path / "custom.db"
    assert s.context_budget_chars == 1234


def test_schema_init_and_fts_sync(tmp_path):
    conn = connect(tmp_path / "nested" / "memory.db")  # parent dirs auto-created
    try:
        init_schema(conn)
        tables = _tables(conn)
        for name in (
            "nodes",
            "edges",
            "provenance",
            "retrieval_traces",
            "url_cache",
            "nodes_fts",
            "nodes_ai",
            "nodes_ad",
            "nodes_au",
        ):
            assert name in tables, f"missing {name}"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

        cur = conn.execute("INSERT INTO nodes (kind, text) VALUES ('entity', 'sky')")
        sky_id = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO nodes (kind, text, confidence) VALUES ('fact', 'The water appears blue.', 0.9)"
        )
        fact_id = cur.lastrowid
        conn.execute(
            "INSERT INTO edges (src, dst, type) VALUES (?, ?, 'related_to')",
            (sky_id, fact_id),
        )
        conn.execute(
            "INSERT INTO provenance (node_id, source_kind) VALUES (?, 'manual')",
            (fact_id,),
        )

        # FTS5 external-content index is populated by the insert trigger.
        assert _fts_ids(conn, "sky") == [sky_id]
        assert _fts_ids(conn, "blue") == [fact_id]

        # Update keeps FTS in sync.
        conn.execute("UPDATE nodes SET text = 'atmosphere' WHERE id = ?", (sky_id,))
        assert _fts_ids(conn, "sky") == []
        assert _fts_ids(conn, "atmosphere") == [sky_id]

        # Delete removes the FTS row and cascades edges + provenance.
        conn.execute("DELETE FROM nodes WHERE id = ?", (fact_id,))
        assert _fts_ids(conn, "blue") == []
        assert conn.execute("SELECT COUNT(*) FROM provenance").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0

        # rebuild_fts re-syncs the index from source rows after drift.
        conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES ('delete-all')")
        assert _fts_ids(conn, "atmosphere") == []
        rebuild_fts(conn)
        assert _fts_ids(conn, "atmosphere") == [sky_id]
    finally:
        conn.close()
