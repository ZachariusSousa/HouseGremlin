"""Phase 1 tests: core store with dedup/merge, contradiction coexistence, forget cascade.

All LLM verdicts come from a deterministic mock judge (strict JSON strings);
no live model is ever called.
"""

import json

import pytest
from pc_memory.app.db import connect, init_schema
from pc_memory.app.store import (
    add_edge,
    add_fact,
    add_triple,
    attach_provenance,
    forget_node,
    inspect_node,
    parse_verdict,
    stats,
    upsert_entity,
)


def _judge(verdict: str):
    """Mock LLM verdict client: returns strict JSON for any prompt."""

    def judge(prompt: str) -> str:
        return json.dumps({"verdict": verdict})

    return judge


@pytest.fixture()
def conn(tmp_path):
    c = connect(tmp_path / "memory.db")
    init_schema(c)
    yield c
    c.close()


def _fts_match_ids(conn, term: str) -> list[int]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH ?", (term,)
        )
    ]


# ---------------------------------------------------------------------------
# entities + edges
# ---------------------------------------------------------------------------


def test_entity_dedup_exact_normalized(conn):
    first, created = upsert_entity(conn, "Alice")
    second, created2 = upsert_entity(conn, "  alice ")
    assert created is True and created2 is False
    assert first == second
    count = conn.execute("SELECT COUNT(*) FROM nodes WHERE kind = 'entity'").fetchone()[
        0
    ]
    assert count == 1


def test_edge_unique_on_src_dst_type(conn):
    a, _ = upsert_entity(conn, "Alice")
    b, _ = upsert_entity(conn, "Bob")
    add_edge(conn, a, b, "knows")
    add_edge(conn, a, b, "knows")  # re-add is a no-op
    rows = conn.execute(
        "SELECT id FROM edges WHERE src = ? AND dst = ? AND type = 'knows'", (a, b)
    ).fetchall()
    assert len(rows) == 1


def test_attach_provenance(conn):
    node_id, _ = upsert_entity(conn, "Alice")
    row_id = attach_provenance(conn, node_id, source_kind="manual", snippet="hello")
    row = conn.execute("SELECT * FROM provenance WHERE id = ?", (row_id,)).fetchone()
    assert row["node_id"] == node_id and row["snippet"] == "hello"


# ---------------------------------------------------------------------------
# fact dedup / merge
# ---------------------------------------------------------------------------


def test_fact_duplicate_merges_into_oldest(conn):
    first = add_fact(conn, "The sky is blue.", confidence=0.5, judge=_judge("new"))
    second = add_fact(
        conn,
        "The sky looks blue.",  # shares FTS terms -> candidate of the first fact
        confidence=0.9,
        provenance={"source_kind": "manual", "snippet": "re-observed"},
        judge=_judge("duplicate"),
    )
    assert second.verdict == "duplicate"
    assert second.node_id == first.node_id  # keep oldest id
    row = conn.execute(
        "SELECT text, confidence FROM nodes WHERE id = ?", (first.node_id,)
    ).fetchone()
    assert row["text"] == "The sky is blue."  # never silent overwrite
    assert row["confidence"] == pytest.approx(0.9)  # raised to max
    prov = conn.execute(
        "SELECT COUNT(*) FROM provenance WHERE node_id = ?", (first.node_id,)
    ).fetchone()[0]
    assert prov == 2  # original + merged provenance


def test_fact_merge_never_lowers_confidence(conn):
    add_fact(conn, "The lake is deep.", confidence=0.9, judge=_judge("new"))
    result = add_fact(
        conn, "The lake is very deep.", confidence=0.2, judge=_judge("duplicate")
    )
    row = conn.execute(
        "SELECT confidence FROM nodes WHERE id = ?", (result.node_id,)
    ).fetchone()
    # Merging a lower-confidence repeat must not lower the stored confidence.
    assert row["confidence"] == pytest.approx(0.9)


def test_fact_without_judge_exact_match_still_merges(conn):
    first = add_fact(conn, "The roof is red.", confidence=0.6)
    second = add_fact(conn, "the roof is red.", confidence=0.7)  # normalized exact
    assert second.verdict == "duplicate" and second.node_id == first.node_id


def test_fact_without_judge_inserts_new(conn):
    result = add_fact(conn, "The garden smells of roses.", confidence=0.5)
    assert result.verdict == "new"
    row = conn.execute(
        "SELECT COUNT(*) FROM provenance WHERE node_id = ?", (result.node_id,)
    ).fetchone()[0]
    assert row >= 1  # every fact carries >=1 provenance row


# ---------------------------------------------------------------------------
# contradiction coexistence (correction 3)
# ---------------------------------------------------------------------------


def test_contradiction_keeps_both_facts_active(conn):
    first = add_fact(conn, "The sky is blue.", confidence=0.8, judge=_judge("new"))
    second = add_fact(
        conn, "The sky is green.", confidence=0.7, judge=_judge("contradiction")
    )
    assert second.verdict == "contradiction"
    assert second.node_id != first.node_id

    # Both fact nodes survive, unchanged in confidence (no age-based change).
    confs = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT id, confidence FROM nodes WHERE kind = 'fact'"
        ).fetchall()
    }
    assert confs[first.node_id] == pytest.approx(0.8)
    assert confs[second.node_id] == pytest.approx(0.7)

    # Linked by contradicted_by edges in both directions.
    edges = {
        (row[0], row[1])
        for row in conn.execute(
            "SELECT src, dst FROM edges WHERE type = 'contradicted_by'"
        ).fetchall()
    }
    assert (first.node_id, second.node_id) in edges
    assert (second.node_id, first.node_id) in edges


# ---------------------------------------------------------------------------
# triples: fact node is the canonical claim (correction 5)
# ---------------------------------------------------------------------------


def test_triple_links_entities_to_canonical_fact(conn):
    result = add_triple(
        conn,
        "Alice",
        "lives_in",
        "Seattle",
        sentence="Alice lives in Seattle.",
        provenance={"source_kind": "manual", "snippet": "bio"},
        judge=_judge("new"),
    )
    assert result.verdict == "new"
    info = inspect_node(conn, result.node_id)
    assert info is not None
    node = info["node"]
    assert node["kind"] == "fact" and node["text"] == "Alice lives in Seattle."
    links = {e["type"]: (e["src_text"], e["dst_text"]) for e in info["edges"]}
    assert links["lives_in"] == ("Alice", "Alice lives in Seattle.")
    assert links["object_of"] == ("Seattle", "Alice lives in Seattle.")
    assert len(info["provenance"]) >= 1


# ---------------------------------------------------------------------------
# forget cascade + inspect/stats
# ---------------------------------------------------------------------------


def test_forget_cascade_leaves_fts_consistent(conn):
    result = add_fact(
        conn,
        "The volcano erupted at dawn.",
        confidence=0.6,
        provenance={"source_kind": "manual", "snippet": "news"},
        judge=_judge("new"),
    )
    other = add_fact(conn, "The river runs cold.", judge=_judge("new"))
    assert _fts_match_ids(conn, "volcano") == [result.node_id]

    assert forget_node(conn, result.node_id) is True

    # Node, its edges and provenance are gone.
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE id = ?", (result.node_id,)
        ).fetchone()[0]
        == 0
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM edges WHERE src = ? OR dst = ?",
            (result.node_id, result.node_id),
        ).fetchone()[0]
        == 0
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM provenance WHERE node_id = ?", (result.node_id,)
        ).fetchone()[0]
        == 0
    )
    # FTS row removed; surviving nodes still searchable.
    assert _fts_match_ids(conn, "volcano") == []
    assert _fts_match_ids(conn, "river") == [other.node_id]
    # Index size matches source rows (no orphan FTS entries).
    fts_count = conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0]
    node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert fts_count == node_count


def test_forget_missing_node_returns_false(conn):
    assert forget_node(conn, 9999) is False


def test_inspect_and_stats(conn):
    add_triple(conn, "Alice", "lives_in", "Seattle", judge=_judge("new"))
    s = stats(conn)
    assert s["nodes"] == 3  # 2 entities + 1 fact
    assert s["nodes_by_kind"] == {"entity": 2, "fact": 1}
    assert s["edges"] == 2
    assert s["provenance"] >= 1

    missing = inspect_node(conn, 9999)
    assert missing is None


# ---------------------------------------------------------------------------
# verdict parsing (strict JSON contract for the LLM judge)
# ---------------------------------------------------------------------------


def test_parse_verdict_accepts_strict_json_and_plain_string():
    assert parse_verdict('{"verdict": "duplicate"}') == "duplicate"
    assert parse_verdict('{"verdict": "CONTRADICTION"}') == "contradiction"
    assert parse_verdict('"new"') == "new"


def test_parse_verdict_rejects_garbage():
    assert parse_verdict("not json") is None
    assert parse_verdict('{"verdict": "maybe"}') is None
    assert parse_verdict("") is None
