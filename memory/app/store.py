"""Core knowledge-graph store: upserts, dedup/merge, forget cascade, inspect/stats.

Rules enforced here (see TODO.md mandatory corrections):
- Fact nodes are canonical claims: every claim is a `fact` node with a sentence,
  confidence and >=1 provenance row. Triple edges link entity nodes to the fact
  node; edges never carry claims by themselves.
- Contradictory facts coexist: a `contradiction` verdict links both fact nodes
  with `contradicted_by` edges and keeps both active. Confidence is never lowered
  (or raised) merely because a fact is older.
- Never silent overwrite: duplicates merge into the oldest node; everything else
  inserts a new node.

The LLM verdict client is injected as `judge(prompt: str) -> str` returning raw
LLM text; this module builds the strict-JSON prompt and parses the verdict, so
unit tests can pass a deterministic mock. The real OpenAI-compatible adapter
lands in Phase 2 (`app/llm.py`).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

VERDICTS = ("duplicate", "related", "contradiction", "new")
EMBEDDING_COSINE_THRESHOLD = 0.9
MAX_CANDIDATES = 5


@dataclass(frozen=True)
class FactAdd:
    """Result of adding a fact. For `duplicate`, node_id is the kept (oldest) id."""

    node_id: int
    verdict: str


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def _words(text: str, max_words: int = 8) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", _norm(text)) if len(w) > 2][:max_words]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity; returns 0.0 for zero-length or corrupt vectors."""
    try:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        score = float(np.dot(a, b) / (na * nb))
    except (TypeError, ValueError):
        return 0.0
    return score if np.isfinite(score) else 0.0


def _find_exact(conn: sqlite3.Connection, kind: str, text: str) -> int | None:
    """Exact normalized-text match within one node kind (MVP: linear scan)."""
    key = _norm(text)
    for row in conn.execute("SELECT id, text FROM nodes WHERE kind = ?", (kind,)):
        if _norm(row["text"]) == key:
            return row["id"]
    return None


def _fact_candidates(
    conn: sqlite3.Connection,
    text: str,
    embedding: Sequence[float] | None = None,
    limit: int = MAX_CANDIDATES,
) -> list[tuple[int, str]]:
    """Existing fact nodes similar to `text`: FTS5 term overlap + cosine when embeddings exist."""
    found: dict[int, str] = {}
    words = _words(text)
    if words:
        match = " OR ".join(f'"{w}"' for w in words)
        try:
            rows = conn.execute(
                "SELECT n.id AS id, n.text AS text FROM nodes_fts f "
                "JOIN nodes n ON n.id = f.rowid "
                'WHERE f.nodes_fts MATCH ? AND n.kind = "fact" LIMIT ?',
                (match, limit * 3),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        for row in rows:
            found.setdefault(row["id"], row["text"])
    if embedding is not None:
        vec = np.asarray(list(embedding), dtype=np.float32)
        stored = conn.execute(
            "SELECT id, text, embedding FROM nodes WHERE kind = 'fact' AND embedding IS NOT NULL"
        ).fetchall()
        for row in stored:
            if row["id"] in found:
                continue
            sim = _cosine(vec, np.frombuffer(row["embedding"], dtype=np.float32))
            if sim >= EMBEDDING_COSINE_THRESHOLD:
                found[row["id"]] = row["text"]
    return list(found.items())[:limit]


def _as_provenance_rows(provenance: Any, default_snippet: str) -> list[dict]:
    if provenance is None:
        return [{"source_kind": "manual", "snippet": default_snippet}]
    if isinstance(provenance, dict):
        rows = [provenance]
    else:
        rows = list(provenance)
    return rows or [{"source_kind": "manual", "snippet": default_snippet}]


# ---------------------------------------------------------------------------
# verdict prompt / parsing (strict JSON)
# ---------------------------------------------------------------------------


def build_verdict_prompt(existing: str, candidate: str) -> str:
    return (
        "You are a memory dedup judge. Compare two stored facts.\n"
        f"EXISTING: {existing}\n"
        f"CANDIDATE: {candidate}\n"
        "Return strict JSON only, no other text, one of:\n"
        '{"verdict": "duplicate"}      (same claim, rephrased)\n'
        '{"verdict": "related"}        (different but topically connected claims)\n'
        '{"verdict": "contradiction"}  (claims that cannot both be true)\n'
        '{"verdict": "new"}            (unrelated / independent claim)'
    )


def parse_verdict(raw: str) -> str | None:
    """Parse strict-JSON verdict output; returns one of VERDICTS or None if invalid."""
    try:
        data = json.loads(raw.strip())
    except (json.JSONDecodeError, AttributeError):
        return None
    if isinstance(data, dict):
        value = data.get("verdict")
    elif isinstance(data, str):
        value = data
    else:
        return None
    value = str(value).strip().lower()
    return value if value in VERDICTS else None


# ---------------------------------------------------------------------------
# upserts
# ---------------------------------------------------------------------------


def upsert_entity(
    conn: sqlite3.Connection, text: str, *, confidence: float = 0.5
) -> tuple[int, bool]:
    """Insert or reuse an entity node by exact normalized text. Returns (id, created)."""
    existing = _find_exact(conn, "entity", text)
    if existing is not None:
        return existing, False
    cur = conn.execute(
        "INSERT INTO nodes (kind, text, confidence) VALUES ('entity', ?, ?)",
        (str(text).strip(), confidence),
    )
    conn.commit()
    new_id = cur.lastrowid
    assert new_id is not None  # sqlite3 sets lastrowid after INSERT
    return new_id, True


def attach_provenance(
    conn: sqlite3.Connection,
    node_id: int,
    source_kind: str = "manual",
    source_ref: str | None = None,
    snippet: str | None = None,
    confidence: float | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO provenance (node_id, source_kind, source_ref, snippet, confidence) "
        "VALUES (?, ?, ?, ?, ?)",
        (node_id, source_kind, source_ref, snippet, confidence),
    )
    conn.commit()
    row_id = cur.lastrowid
    assert row_id is not None  # sqlite3 sets lastrowid after INSERT
    return row_id


def add_edge(
    conn: sqlite3.Connection, src: int, dst: int, edge_type: str
) -> int | None:
    """Insert a typed edge; (src, dst, type) is unique so re-adding is a no-op."""
    conn.execute(
        "INSERT OR IGNORE INTO edges (src, dst, type) VALUES (?, ?, ?)",
        (src, dst, edge_type),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM edges WHERE src = ? AND dst = ? AND type = ?",
        (src, dst, edge_type),
    ).fetchone()
    return row["id"] if row else None


def add_fact(
    conn: sqlite3.Connection,
    text: str,
    *,
    confidence: float = 0.5,
    provenance: dict | list[dict] | None = None,
    judge: Callable[[str], str] | None = None,
    embedding: Sequence[float] | None = None,
) -> FactAdd:
    """Add a fact node with dedup/merge via exact match, FTS5 and (optionally) embeddings.

    With `judge` (LLM verdict client), each similar existing fact is judged
    duplicate/related/contradiction/new. Duplicate merges into the oldest node
    (confidence raised to the max, provenance added); contradiction keeps both
    facts active linked by `contradicted_by` edges; related links with
    `related_to`; new inserts a standalone fact. Without `judge`, exact matches
    still merge and everything else is inserted as new. Never overwrites silently.
    """
    sentence = str(text).strip()
    rows = _as_provenance_rows(provenance, sentence)

    def merge(existing_id: int) -> FactAdd:
        old_conf = conn.execute(
            "SELECT confidence FROM nodes WHERE id = ?", (existing_id,)
        ).fetchone()[0]
        try:
            merged_conf = max(float(old_conf), float(confidence))
        except (TypeError, ValueError):
            merged_conf = float(confidence)  # corrupt stored value: keep the new one
        conn.execute(
            "UPDATE nodes SET confidence = ?, updated_at = datetime('now') WHERE id = ?",
            (merged_conf, existing_id),
        )
        for p in rows:
            attach_provenance(conn, existing_id, **p)
        return FactAdd(existing_id, "duplicate")

    exact = _find_exact(conn, "fact", sentence)
    if exact is not None:
        return merge(exact)

    embedding_blob = (
        np.asarray(list(embedding), dtype=np.float32).tobytes()
        if embedding is not None
        else None
    )

    candidates = _fact_candidates(conn, sentence, embedding)
    for cand_id, cand_text in candidates:
        if judge is None:
            break
        verdict = parse_verdict(judge(build_verdict_prompt(cand_text, sentence)))
        if verdict == "duplicate":
            return merge(cand_id)
        if verdict in ("contradiction", "related"):
            cur = conn.execute(
                "INSERT INTO nodes (kind, text, confidence, embedding) VALUES ('fact', ?, ?, ?)",
                (sentence, confidence, embedding_blob),
            )
            new_id = cur.lastrowid
            assert new_id is not None  # sqlite3 sets lastrowid after INSERT
            for p in rows:
                attach_provenance(conn, new_id, **p)
            if verdict == "contradiction":
                add_edge(conn, new_id, cand_id, "contradicted_by")
                add_edge(conn, cand_id, new_id, "contradicted_by")
            else:
                add_edge(conn, new_id, cand_id, "related_to")
            return FactAdd(new_id, verdict)

    cur = conn.execute(
        "INSERT INTO nodes (kind, text, confidence, embedding) VALUES ('fact', ?, ?, ?)",
        (sentence, confidence, embedding_blob),
    )
    new_id = cur.lastrowid
    assert new_id is not None  # sqlite3 sets lastrowid after INSERT
    for p in rows:
        attach_provenance(conn, new_id, **p)
    return FactAdd(new_id, "new")


def add_triple(
    conn: sqlite3.Connection,
    subject: str,
    predicate: str,
    object_: str,
    *,
    sentence: str | None = None,
    confidence: float = 0.5,
    provenance: dict | list[dict] | None = None,
    judge: Callable[[str], str] | None = None,
    embedding: Sequence[float] | None = None,
) -> FactAdd:
    """Add a subject–predicate–object triple as the canonical fact node plus
    typed edges from both entity nodes to the fact node (correction 5)."""
    subj_id, _ = upsert_entity(conn, subject)
    obj_id, _ = upsert_entity(conn, object_)
    sentence = sentence or f"{subject} {predicate} {object_}."
    result = add_fact(
        conn,
        sentence,
        confidence=confidence,
        provenance=provenance,
        judge=judge,
        embedding=embedding,
    )
    add_edge(conn, subj_id, result.node_id, predicate)
    add_edge(conn, obj_id, result.node_id, "object_of")
    return result


# ---------------------------------------------------------------------------
# forget cascade + inspection
# ---------------------------------------------------------------------------


def forget_node(conn: sqlite3.Connection, node_id: int) -> bool:
    """Delete a node and everything attached to it (edges both directions,
    provenance). FTS5 rows are removed by the nodes_ad trigger. Returns True if
    the node existed."""
    conn.execute("DELETE FROM edges WHERE src = ? OR dst = ?", (node_id, node_id))
    conn.execute("DELETE FROM provenance WHERE node_id = ?", (node_id,))
    cur = conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
    conn.commit()
    return cur.rowcount > 0


def inspect_node(conn: sqlite3.Connection, node_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
    if row is None:
        return None
    edges = conn.execute(
        "SELECT e.id, e.src, e.dst, e.type, ns.text AS src_text, nd.text AS dst_text "
        "FROM edges e JOIN nodes ns ON ns.id = e.src JOIN nodes nd ON nd.id = e.dst "
        "WHERE e.src = ? OR e.dst = ? ORDER BY e.id",
        (node_id, node_id),
    ).fetchall()
    provenance = conn.execute(
        "SELECT * FROM provenance WHERE node_id = ? ORDER BY id", (node_id,)
    ).fetchall()
    return {
        "node": dict(row),
        "edges": [dict(e) for e in edges],
        "provenance": [dict(p) for p in provenance],
    }


_STATS_QUERIES = frozenset(
    {
        "SELECT kind, COUNT(*) FROM nodes GROUP BY kind",
        "SELECT type, COUNT(*) FROM edges GROUP BY type",
    }
)


def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    def counts(sql: str) -> dict:
        if sql not in _STATS_QUERIES:  # allowlist: only the static queries above
            raise ValueError(f"unregistered stats query: {sql!r}")
        return dict(conn.execute(sql).fetchall())

    return {
        "nodes": conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
        "nodes_by_kind": counts("SELECT kind, COUNT(*) FROM nodes GROUP BY kind"),
        "edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        "edges_by_type": counts("SELECT type, COUNT(*) FROM edges GROUP BY type"),
        "provenance": conn.execute("SELECT COUNT(*) FROM provenance").fetchone()[0],
        "retrieval_traces": conn.execute(
            "SELECT COUNT(*) FROM retrieval_traces"
        ).fetchone()[0],
    }


def re_embed(conn: sqlite3.Connection, embed) -> dict[str, Any]:
    """Re-embed every node with the current embedding model.

    Needed whenever the embedding backend changes (e.g. switching from one
    model to another): vectors from different models are not comparable, so
    all stored blobs must be regenerated in one pass. Returns a summary.
    """
    rows = conn.execute("SELECT id, text FROM nodes ORDER BY id").fetchall()
    batch_size = 32
    updated = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        vectors = embed.embed([r["text"] for r in chunk])
        if vectors is None or len(vectors) != len(chunk):
            return {
                "embedded": updated,
                "total": len(rows),
                "failed_at_batch": start // batch_size,
            }
        for row, vec in zip(chunk, vectors):
            blob = np.asarray(list(vec), dtype=np.float32).tobytes()
            conn.execute(
                "UPDATE nodes SET embedding = ?, updated_at = datetime('now') WHERE id = ?",
                (blob, row["id"]),
            )
            updated += 1
    conn.commit()
    return {"embedded": updated, "total": len(rows), "failed_at_batch": None}
