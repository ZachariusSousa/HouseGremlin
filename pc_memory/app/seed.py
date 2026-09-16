"""Seed corpus: a hand-built "why is the sky blue" causal chain + distractors.

Idempotent: facts dedup by exact text (store.add_fact) and edges are
UNIQUE(src, dst, type), so re-running never duplicates anything. No LLM needed.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from pc_memory.app.store import add_edge, add_fact

# Causal chain: sky -> Rayleigh scattering -> short-wavelength scattering.
CHAIN: list[tuple[str, str]] = [
    ("sky_appears_blue", "The sky appears blue because of Rayleigh scattering."),
    (
        "rayleigh_scattering",
        "Rayleigh scattering is the scattering of light by particles much smaller than its wavelength, such as air molecules.",
    ),
    (
        "short_wavelengths_scatter_more",
        "In Rayleigh scattering, shorter wavelengths scatter much more strongly, with intensity proportional to 1/lambda^4.",
    ),
    ("blue_light_is_shorter", "Blue light has a shorter wavelength than red light."),
    (
        "scattered_light_reaches_eyes",
        "So the scattered sunlight reaching our eyes from every direction is predominantly blue.",
    ),
]

# (src index, dst index, edge type) — direction follows the explanation.
CHAIN_EDGES: list[tuple[int, int, str]] = [
    (0, 1, "explains"),
    (1, 2, "detail"),
    (2, 3, "detail"),
    (3, 4, "consequence"),
]

DISTRACTORS: list[str] = [
    "An octopus has three hearts.",
    "Bananas are slightly radioactive because of potassium-40.",
    "The Eiffel Tower is taller in summer because heat expands the metal.",
]


def _embed_texts(embed: Any, texts: list[str]) -> dict[str, list[float]]:
    """Batch-embed when available; {} when embeddings are unavailable."""
    if embed is None or not texts:
        return {}
    try:
        vectors = embed.embed(list(texts))
    except Exception:  # noqa: BLE001 - embeddings are optional
        return {}
    if not vectors:
        return {}
    return dict(zip(texts, vectors, strict=True))


def seed_sky_chain(
    conn: sqlite3.Connection, *, embed: Any = None, confidence: float = 0.9
) -> dict[str, Any]:
    """Insert the causal chain + distractors; safe to run repeatedly."""
    sentences = [text for _label, text in CHAIN] + list(DISTRACTORS)
    vectors = _embed_texts(embed, sentences)

    chain_ids: list[int] = []
    for _label, sentence in CHAIN:
        result = add_fact(
            conn,
            sentence,
            confidence=confidence,
            provenance={
                "source_kind": "manual",
                "source_ref": "seed-script",
                "snippet": sentence[:500],
            },
            embedding=vectors.get(sentence),
        )
        chain_ids.append(result.node_id)

    for src_idx, dst_idx, edge_type in CHAIN_EDGES:
        add_edge(conn, chain_ids[src_idx], chain_ids[dst_idx], edge_type)

    distractor_ids: list[int] = []
    for sentence in DISTRACTORS:
        result = add_fact(
            conn,
            sentence,
            confidence=confidence,
            provenance={
                "source_kind": "manual",
                "source_ref": "seed-script",
                "snippet": sentence[:500],
            },
            embedding=vectors.get(sentence),
        )
        distractor_ids.append(result.node_id)

    return {
        "chain": dict(zip((label for label, _ in CHAIN), chain_ids, strict=True)),
        "distractors": distractor_ids,
    }
