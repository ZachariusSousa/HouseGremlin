"""Ingestion orchestration: LLM extraction -> canonical store writes.

Shared by the FastAPI endpoints, the CLI and the tests. All-or-nothing per
source text: when the LLM extraction is invalid (`ExtractionError`) the caller
performs no store writes at all (never partial writes).

Provenance rule: every ingested fact/triple carries a provenance row with
`source_kind` + `source_ref` (URL for web, owner reference for manual text) so
every durable memory retains its source.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from pc_memory.app.extract import extract_facts
from pc_memory.app.search import fetch_url
from pc_memory.app.store import add_fact, add_triple

logger = logging.getLogger("pc_memory.ingest")


class IngestError(RuntimeError):
    """Raised when a source cannot be ingested (e.g. URL fetch failed)."""


def _judge_from(llm: Any):
    """Store verdict client from an LLM object (`chat(prompt) -> str`), or None."""
    if llm is None:
        return None
    return getattr(llm, "chat", None)


def _embed_map(embed: Any, texts: Sequence[str]) -> dict[str, list[float]]:
    """Batch-embed texts; returns {text: vector} (empty when embeddings unavailable)."""
    if embed is None or not texts:
        return {}
    vectors = embed.embed(list(texts))
    if vectors is None or len(vectors) != len(texts):
        return {}
    return dict(zip((str(t) for t in texts), vectors, strict=True))


def ingest_text(
    conn,
    text: str,
    *,
    llm: Any = None,
    embed: Any = None,
    confidence: float = 0.7,
    source_kind: str = "manual",
    source_ref: str | None = None,
) -> dict[str, Any]:
    """Extract facts from `text` and store them as canonical fact nodes.

    Returns a summary: {"triples": [...], "facts": [...], "node_ids": [...]}.
    Raises ExtractionError when the LLM output is invalid (nothing written).
    """
    body = str(text or "").strip()
    if not body:
        raise IngestError("text is empty")

    extracted = extract_facts(llm, body)  # may raise ExtractionError -> no writes
    claims = [t.claim for t in extracted.triples] + list(extracted.facts)
    vectors = _embed_map(embed, claims)
    judge = _judge_from(llm)
    provenance = {
        "source_kind": source_kind,
        "source_ref": source_ref,
        "snippet": body[:500],
    }

    node_ids: list[int] = []
    triples_out: list[dict[str, Any]] = []
    for triple in extracted.triples:
        result = add_triple(
            conn,
            triple.subject,
            triple.predicate,
            triple.object,
            sentence=triple.claim,
            confidence=confidence,
            provenance=provenance,
            judge=judge,
            embedding=vectors.get(triple.claim),
        )
        node_ids.append(result.node_id)
        triples_out.append(
            {
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object,
                "node_id": result.node_id,
                "verdict": result.verdict,
            }
        )

    facts_out: list[dict[str, Any]] = []
    for sentence in extracted.facts:
        result = add_fact(
            conn,
            sentence,
            confidence=confidence,
            provenance=provenance,
            judge=judge,
            embedding=vectors.get(sentence),
        )
        node_ids.append(result.node_id)
        facts_out.append(
            {"text": sentence, "node_id": result.node_id, "verdict": result.verdict}
        )

    return {"triples": triples_out, "facts": facts_out, "node_ids": node_ids}


def ingest_url(
    conn,
    url: str,
    *,
    llm: Any = None,
    embed: Any = None,
    confidence: float = 0.7,
) -> dict[str, Any]:
    """Fetch `url` (cached in url_cache) and run it through ingest_text.

    Raises IngestError when the page cannot be fetched (skip-and-continue at a
    higher level is the caller's choice; a single /ingest/url call fails loud).
    """
    url = str(url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise IngestError(f"not an http(s) url: {url!r}")
    text = fetch_url(conn, url)
    if not text:
        raise IngestError(f"fetch failed or empty page: {url}")
    return ingest_text(
        conn,
        text,
        llm=llm,
        embed=embed,
        confidence=confidence,
        source_kind="web",
        source_ref=url,
    )
