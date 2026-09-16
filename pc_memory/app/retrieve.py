"""Routed retrieval with trace logging.

Pipeline: hybrid seeds (FTS5 bm25 + embedding cosine, fused via RRF) ->
LLM hop router walking typed edges (bounded hops/beam) -> budgeted context
assembly (hard char budget, lowest-relevance tail dropped) -> trace row in
`retrieval_traces`.

The LLM is only consulted for hop decisions (`complete_json`, strict JSON);
all ranking math is deterministic and unit-testable without any network.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from pc_memory.app.llm import extract_json_object

RRF_K = 60
_SEED_LIMIT = 20

# Function words that carry no retrieval signal; "is"/"the" would otherwise
# match nearly every node and pollute the seed set.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "did",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "is",
        "it",
        "its",
        "may",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
    }
)


@dataclass
class HopDecision:
    hop: int
    candidates: list[int]
    chosen: list[int]
    reasons: dict[int, str] = field(default_factory=dict)
    stop: bool = False


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def _rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal-rank fusion over one or more ranked id lists (best-first)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, node_id in enumerate(ranking):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)
    return scores


def fts_seed_ids(
    conn: sqlite3.Connection, query: str, limit: int = _SEED_LIMIT
) -> list[int]:
    """BM25-ranked node ids matching any significant query token (FTS5 index).

    Stopwords are dropped first so function words like "is"/"the" cannot pull
    unrelated nodes into the seed set.
    """
    tokens = [
        t
        for t in re.findall(r"[a-z0-9]+", query.lower())
        if len(t) > 1 and t not in _STOPWORDS
    ]
    if not tokens:
        return []
    match = " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))
    try:
        rows = conn.execute(
            "SELECT n.id AS id FROM nodes_fts f JOIN nodes n ON n.id = f.rowid "
            "WHERE f.nodes_fts MATCH ? ORDER BY bm25(nodes_fts) LIMIT ?",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []  # malformed match expression -> no FTS seeds, embeddings still work
    return [row["id"] for row in rows]


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cosine similarity with a non-finite guard; None when undefined."""
    try:
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))
        if a_norm == 0.0 or b_norm == 0.0:
            return None
        sim = float(np.dot(a, b) / (a_norm * b_norm))
    except (FloatingPointError, ValueError):
        return None
    if not np.isfinite(sim):
        return None
    return sim


def embedding_seed_ids(
    conn: sqlite3.Connection, embed: Any, question: str, limit: int = _SEED_LIMIT
) -> list[int]:
    """Cosine-similarity-ranked node ids against the question vector."""
    if embed is None:
        return []
    try:
        vectors = embed.embed([question])
    except Exception:  # noqa: BLE001 - embedding failure degrades to FTS-only
        return []
    if not vectors:
        return []
    qv = np.asarray(vectors[0], dtype=np.float32)
    rows = conn.execute(
        "SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL"
    ).fetchall()
    scored: list[tuple[float, int]] = []
    for row in rows:
        vec = np.frombuffer(row["embedding"], dtype=np.float32)
        if vec.shape != qv.shape:
            continue
        sim = _cosine(qv, vec)
        if sim is not None:
            scored.append((sim, row["id"]))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [node_id for _, node_id in scored[:limit]]


def seed_nodes(
    conn: sqlite3.Connection,
    question: str,
    *,
    embed: Any = None,
    limit: int = _SEED_LIMIT,
) -> list[int]:
    """Hybrid seeds: FTS5 + embedding rankings fused with RRF (best-first)."""
    rankings = [
        r
        for r in (
            fts_seed_ids(conn, question, limit),
            embedding_seed_ids(conn, embed, question, limit),
        )
        if r
    ]
    if not rankings:
        return []
    fused = _rrf_fuse(rankings)
    return [
        node_id for node_id, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    ][:limit]


# ---------------------------------------------------------------------------
# Hop router
# ---------------------------------------------------------------------------


def _node_text(conn: sqlite3.Connection, node_id: int) -> str | None:
    row = conn.execute("SELECT text FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return row["text"] if row is not None else None


def _frontier(conn: sqlite3.Connection, visited: list[int]) -> list[tuple[int, str]]:
    """Unvisited edge-neighbors of the visited set as (id, edge_type) pairs."""
    visited_set = set(visited)
    if not visited_set:
        return []
    placeholders = ",".join("?" for _ in visited_set)
    rows = conn.execute(
        f"SELECT src, dst, type FROM edges WHERE src IN ({placeholders}) OR dst IN ({placeholders})",
        (*visited, *visited),
    ).fetchall()
    seen: dict[int, str] = {}
    for row in rows:
        other = row["dst"] if row["src"] in visited_set else row["src"]
        if other not in visited_set and other not in seen:
            seen[other] = row["type"]
    return list(seen.items())


def _similar_unvisited(
    conn: sqlite3.Connection,
    embed: Any,
    question: str,
    visited: list[int],
    limit: int = 3,
) -> list[int]:
    """Embedding-similar unvisited nodes (hybrid candidate expansion)."""
    if embed is None:
        return []
    visited_set = set(visited)
    ids = embedding_seed_ids(conn, embed, question, max(limit * 2, limit))
    out: list[int] = []
    for node_id in ids:
        if node_id not in visited_set and len(out) < limit:
            out.append(node_id)
    return out


def build_router_prompt(
    question: str,
    visited: list[tuple[int, str]],
    candidates: list[tuple[int, str, str]],
) -> str:
    """Strict-JSON hop prompt. Candidates are (id, via, text); visited is (id, text)."""
    lines = [
        "You are a knowledge-graph retrieval router.",
        f"QUESTION: {question}",
        "VISITED (already in context):",
    ]
    lines += [f"- id={node_id}: {text}" for node_id, text in visited] or ["- (none)"]
    lines.append("CANDIDATES (unvisited neighbors / similar nodes):")
    lines += [f"- id={cid} (via {via}): {text}" for cid, via, text in candidates]
    lines.append(
        "Choose the candidate node ids that best help answer the question, "
        "or stop when the visited nodes already answer it. Return strict JSON only: "
        '{"stop": true} or {"choose": [{"id": <int>, "reason": "<one line>"}]}'
    )
    return "\n".join(lines)


def parse_hop_decision(
    raw: Any, candidate_ids: list[int]
) -> tuple[list[int], dict[int, str], bool]:
    """Parse a router reply into (chosen ids, reasons, stop). Unknown ids are dropped."""
    data = raw if isinstance(raw, dict) else extract_json_object(raw)
    if not isinstance(data, dict):
        return [], {}, True  # unparseable -> stop rather than loop forever
    allowed = set(candidate_ids)
    chosen: list[int] = []
    reasons: dict[int, str] = {}
    items = data.get("choose") or []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_id = item.get("id")
            if isinstance(raw_id, bool) or not isinstance(raw_id, (int, float, str)):
                continue
            try:
                node_id = (
                    int(raw_id.strip()) if isinstance(raw_id, str) else int(raw_id)
                )
            except ValueError:
                continue
            if node_id in allowed and node_id not in chosen:
                chosen.append(node_id)
                reasons[node_id] = str(item.get("reason") or "")[:200]
    stop = bool(data.get("stop"))
    return chosen, reasons, stop


# ---------------------------------------------------------------------------
# Context assembly + trace
# ---------------------------------------------------------------------------


def _node_links(conn: sqlite3.Connection, node_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT e.type AS type, e.src AS src, n.text AS other FROM edges e "
        "JOIN nodes n ON n.id = CASE WHEN e.src = ? THEN e.dst ELSE e.src END "
        "WHERE e.src = ? OR e.dst = ?",
        (node_id, node_id, node_id),
    ).fetchall()
    links: list[str] = []
    for row in rows:
        direction = "->" if row["src"] == node_id else "<-"
        other = re.sub(r"\s+", " ", row["other"])[:60]
        links.append(f"{row['type']}{direction}{other}")
    return links


def assemble_context(
    conn: sqlite3.Connection, visited: list[int], budget_chars: int
) -> tuple[str, list[dict[str, Any]]]:
    """Render visited nodes (visit order == relevance order) under a hard char budget.

    Returns (context, included_blocks); lowest-relevance tail is dropped first.
    """
    blocks: list[dict[str, Any]] = []
    for node_id in visited:
        row = conn.execute(
            "SELECT id, kind, text FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            continue
        blocks.append(
            {
                "id": node_id,
                "kind": row["kind"],
                "text": row["text"],
                "links": _node_links(conn, node_id),
            }
        )
    lines: list[str] = []
    total = 0
    for idx, block in enumerate(blocks, start=1):
        line = f"{idx}. [{block['kind']}] {block['text']}"
        if block["links"]:
            line += " | " + "; ".join(block["links"])
        if lines and total + len(line) > budget_chars:
            break  # drop this and all lower-relevance blocks
        lines.append(line)
        total += len(line) + 1
    included = blocks[: len(lines)] if lines else []
    return "\n".join(lines), included


def _write_trace(
    conn: sqlite3.Connection,
    question: str,
    seed_ids: list[int],
    decisions: list[HopDecision],
    visited: list[int],
    context: str,
) -> int:
    cur = conn.execute(
        "INSERT INTO retrieval_traces (question, seed_ids, decisions, visited_ids, context) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            question,
            json.dumps(seed_ids),
            json.dumps([asdict(d) for d in decisions]),
            json.dumps(visited),
            context,
        ),
    )
    conn.commit()
    trace_id = cur.lastrowid
    assert isinstance(trace_id, int) and trace_id > 0
    return trace_id


def get_trace(conn: sqlite3.Connection, trace_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM retrieval_traces WHERE id = ?", (trace_id,)
    ).fetchone()
    if row is None:
        return None
    data = dict(row)
    for key in ("seed_ids", "decisions", "visited_ids"):
        raw_value = data[key]
        try:
            data[key] = json.loads(raw_value) if isinstance(raw_value, str) else []
        except ValueError:
            data[key] = []
    return data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def retrieve(
    conn: sqlite3.Connection,
    question: str,
    *,
    llm: Any = None,
    embed: Any = None,
    max_hops: int | None = None,
    beam: int | None = None,
    budget_chars: int | None = None,
) -> dict[str, Any]:
    """Answer-oriented retrieval over the graph.

    Seeds are hybrid (FTS5 + embedding, RRF-fused); when an LLM is supplied it
    routes up to `max_hops` hops over typed edges with a `beam`-wide candidate
    set per hop; context is rendered under a hard char budget and every run is
    traced in `retrieval_traces`. Degrades gracefully: no embed -> FTS-only,
    no llm -> seeds only.
    """
    question = str(question).strip()
    if not question:
        raise ValueError("question is empty")
    max_hops = max(0, _as_int(max_hops if max_hops is not None else 3, 3))
    beam = max(1, _as_int(beam if beam is not None else 6, 6))
    budget_chars = max(1, _as_int(budget_chars if budget_chars else 6000, 6000))

    seed_ids = seed_nodes(conn, question, embed=embed)
    visited: list[int] = []
    for node_id in seed_ids[:beam]:
        if node_id not in visited and _node_text(conn, node_id) is not None:
            visited.append(node_id)

    decisions: list[HopDecision] = []
    if llm is not None and max_hops > 0 and visited:
        for hop in range(1, max_hops + 1):
            frontier = _frontier(conn, visited)
            candidate_ids: list[int] = [cid for cid, _ in frontier][:beam]
            via: dict[int, str] = {
                cid: f"edge:{etype}" for cid, etype in frontier[:beam]
            }
            for node_id in _similar_unvisited(conn, embed, question, visited):
                if len(candidate_ids) >= beam:
                    break
                candidate_ids.append(node_id)
                via[node_id] = "embedding"
            candidates: list[tuple[int, str, str]] = []
            for cid in candidate_ids:
                text = _node_text(conn, cid)
                if text is not None:
                    candidates.append((cid, via[cid], text))
            if not candidates:
                break
            prompt = build_router_prompt(
                question,
                [(nid, _node_text(conn, nid) or "") for nid in visited],
                candidates,
            )
            raw = llm.complete_json(prompt)
            chosen, reasons, stop = parse_hop_decision(
                raw, [cid for cid, _, _ in candidates]
            )
            decisions.append(
                HopDecision(
                    hop=hop,
                    candidates=[cid for cid, _, _ in candidates],
                    chosen=chosen,
                    reasons=reasons,
                    stop=stop,
                )
            )
            for node_id in chosen:
                if node_id not in visited and _node_text(conn, node_id) is not None:
                    visited.append(node_id)
            if stop or not chosen:
                break

    context, included = assemble_context(conn, visited, budget_chars)
    trace_id = _write_trace(
        conn, question, seed_ids[:beam], decisions, visited, context
    )
    return {"context": context, "nodes": included, "trace_id": trace_id}


__all__ = [
    "HopDecision",
    "assemble_context",
    "build_router_prompt",
    "embedding_seed_ids",
    "fts_seed_ids",
    "get_trace",
    "parse_hop_decision",
    "retrieve",
    "seed_nodes",
]
