"""Autonomous topic researcher: grow a subject's knowledge graph from the web.

Give it a subject ("mario kart wii") and, while you're away, it searches the web
for that subject, fetches the top sources, and asks the LLM to extract facts +
triples about the subject AND name related subjects. The related subjects form a
BFS frontier, so the graph grows organically around the seed — a small
Wikipedia-style slice you can route through at retrieval time with Bekko
embeddings, handing a small LLM a tight pre-researched context.

The LLM is required (it finds the relationships); there is no deterministic
fallback. In tests the LLM is mocked; the CLI needs a live LLM endpoint
(MEMORY_LLM_BASE_URL).

State (visited subjects + frontier queue) lives in a JSON file next to the
memory DB, so research resumes where it left off after an interruption.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from pc_memory.app.extract import ExtractionError, Triple
from pc_memory.app.llm import LLMError
from pc_memory.app.search import fetch_url, web_search

logger = logging.getLogger("pc_memory.research")

# How much page text each source contributes to the extraction prompt.
MAX_SOURCE_CHARS = 6000
# Related subjects kept per source (the rest is noise for frontier purposes).
RELATED_PER_SOURCE = 8


class ResearchExtraction(BaseModel):
    """LLM output for one subject: claims about it + related subjects."""

    triples: list[Triple] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)

    @field_validator("facts", "related")
    @classmethod
    def _clean_strings(cls, values: list[str]) -> list[str]:
        return [v.strip() for v in values if v and v.strip()]


RESEARCH_SYSTEM_PROMPT = (
    "You extract knowledge-graph facts from web page text about a subject. "
    'Return only strict JSON matching {"triples":[{"subject":"...",'
    '"predicate":"snake_case_verb","object":"...","sentence":"one declarative '
    'sentence stating the claim"}],"facts":["other important standalone '
    'sentences"],"related":["names of other real subjects this page connects '
    'the main subject to (people, places, series, platforms, companies, games, '
    'works) — short proper names only, max 8]}. '
    "Extract only what the text actually says; never invent entities or claims. "
    "Keep triples and facts about the main subject itself. If nothing "
    'extractable return {"triples":[],"facts":[],"related":[]}.'
)


def build_research_prompt(subject: str, text: str) -> str:
    return (
        f"Extract knowledge-graph facts about {subject!r} from the web page text below.\n\n"
        f'TEXT:\n"""\n{text.strip()}\n"""\n\n'
        'Return only the strict JSON object {"triples":[...],"facts":[...],"related":[...]}'
    )


def validate_research_extraction(data: Any) -> ResearchExtraction:
    return ResearchExtraction.model_validate(data)


def extract_research(llm, subject: str, text: str) -> ResearchExtraction:
    """Run the research extraction prompt through the LLM and validate strictly.

    Raises ExtractionError (logged) when the output is invalid — callers then
    skip that source without any store writes.
    """
    try:
        data = llm.complete_json(
            build_research_prompt(subject, text),
            system=RESEARCH_SYSTEM_PROMPT,
            validate=validate_research_extraction,
        )
    except (LLMError, ValidationError) as exc:
        logger.warning("research extraction rejected for %s (%d chars): %s", subject, len(text), exc)
        raise ExtractionError(f"invalid research extraction output: {exc}") from exc
    return validate_research_extraction(data)


def _claim_to_url(claims: list[str], urls: list[str]) -> dict[str, str]:
    """Round-robin claims across their source URLs for provenance."""
    out: dict[str, str] = {}
    for i, claim in enumerate(claims):
        if urls:
            out[claim] = urls[i % len(urls)]
    return out


def research_subject(
    subject: str,
    conn,
    *,
    llm,
    embed=None,
    confidence: float = 0.6,
    max_sources: int = 3,
) -> dict[str, Any]:
    """Search the web for one subject, fetch top sources, LLM-extract + store.

    Returns a summary including the related-subject frontier to expand next.
    Skip-and-continue: a bad search/fetch/extraction never raises out of here.
    """
    urls = web_search(subject, max_results=max_sources)
    claims: list[str] = []
    triples_out: list[dict[str, str]] = []
    seen_claims: set[str] = set()
    related: list[str] = []
    seen_related: set[str] = set()
    used_urls: list[str] = []

    for url in urls:
        text = fetch_url(conn, url)
        if not text:
            continue
        try:
            extracted = extract_research(llm, subject, text[:MAX_SOURCE_CHARS])
        except ExtractionError:
            logger.warning("skipping source %s (invalid extraction)", url)
            continue
        used_urls.append(url)
        for triple in extracted.triples:
            if triple.claim not in seen_claims:
                seen_claims.add(triple.claim)
                claims.append(triple.claim)
                triples_out.append(
                    {"subject": triple.subject, "predicate": triple.predicate, "object": triple.object}
                )
        for fact in extracted.facts:
            if fact not in seen_claims:
                seen_claims.add(fact)
                claims.append(fact)
        for rel in extracted.related[:RELATED_PER_SOURCE]:
            if rel.lower() != subject.lower() and rel not in seen_related:
                seen_related.add(rel)
                related.append(rel)

    from pc_memory.app.store import add_fact, add_triple  # local: keep module import light

    claim_urls = _claim_to_url(claims, used_urls)
    vectors: dict[str, list[float]] = {}
    if embed is not None and claims:
        vecs = embed.embed(claims)
        if vecs and len(vecs) == len(claims):
            vectors = dict(zip(claims, vecs))

    added = 0
    merged = 0
    for i, claim in enumerate(claims):
        url = claim_urls.get(claim, "")
        provenance = {"source_kind": "web", "source_ref": url, "snippet": claim[:500]}
        if i < len(triples_out):
            t = triples_out[i]
            result = add_triple(
                conn,
                t["subject"],
                t["predicate"],
                t["object"],
                sentence=claim,
                confidence=confidence,
                provenance=provenance,
                embedding=vectors.get(claim),
            )
        else:
            result = add_fact(
                conn, claim, confidence=confidence, provenance=provenance, embedding=vectors.get(claim)
            )
        if result.verdict == "duplicate":
            merged += 1
        else:
            added += 1

    return {
        "subject": subject,
        "sources": len(used_urls),
        "facts_added": added,
        "facts_merged": merged,
        "related_found": len(related),
        "related": related,
    }


@dataclass
class ResearchState:
    """Resumable BFS state for a research session (JSON-serializable)."""

    seed: str
    visited: dict[str, int] = field(default_factory=dict)  # subject -> depth
    frontier: list[str] = field(default_factory=list)
    totals: dict[str, int] = field(
        default_factory=lambda: {"facts_added": 0, "facts_merged": 0}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "visited": self.visited,
            "frontier": self.frontier,
            "totals": self.totals,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchState":
        return cls(
            seed=data.get("seed", ""),
            visited=dict(data.get("visited", {})),
            frontier=list(data.get("frontier", [])),
            totals=dict(data.get("totals", {"facts_added": 0, "facts_merged": 0})),
        )


def load_research_state(path: Path) -> ResearchState | None:
    if not path.exists():
        return None
    try:
        return ResearchState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        logger.warning("research state unreadable (%s); starting fresh", path)
        return None


def save_research_state(path: Path, state: ResearchState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def research_topic(
    seed: str,
    conn,
    *,
    llm,
    state_path: Path,
    embed=None,
    confidence: float = 0.6,
    max_depth: int = 1,
    breadth_per_level: int = 5,
    max_sources: int = 3,
) -> dict[str, Any]:
    """Expand the subject graph from `seed` up to `max_depth`.

    BFS over the related-subject frontier: depth 0 is the seed, each level
    researches at most `breadth_per_level` new subjects. Resumable via
    `state_path`. Returns a summary of everything stored this call.
    """
    state = load_research_state(state_path) or ResearchState(seed=seed)
    # A fresh `research <subject>` call always re-queues its subject; a resume
    # (watch-research, or the same seed already visited with a live frontier)
    # only continues the pending frontier.
    if not state.frontier and seed.lower() not in state.visited:
        state.frontier = [seed]

    researched: list[dict[str, Any]] = []
    depth = 0
    while state.frontier and depth <= max_depth:
        level = state.frontier[:breadth_per_level]
        state.frontier = state.frontier[breadth_per_level:]
        for subject in level:
            key = subject.lower()
            if key in state.visited:
                continue
            try:
                summary = research_subject(
                    subject, conn, llm=llm, embed=embed, confidence=confidence, max_sources=max_sources
                )
            except Exception as exc:  # noqa: BLE001 - skip-and-continue per subject
                logger.warning("research(%s) failed: %s", subject, exc)
                state.visited[key] = depth
                continue
            researched.append(summary)
            state.totals["facts_added"] += summary["facts_added"]
            state.totals["facts_merged"] += summary["facts_merged"]
            for rel in summary["related"]:
                if (
                    rel.lower() not in state.visited
                    and all(rel.lower() != f.lower() for f in state.frontier)
                ):
                    state.frontier.append(rel)
            state.visited[key] = depth
        depth += 1

    save_research_state(state_path, state)

    return {
        "seed": seed,
        "levels_explored": min(depth, max_depth + 1),
        "subjects_researched": len(researched),
        "visited_total": len(state.visited),
        "frontier_remaining": len(state.frontier),
        "totals": state.totals,
        "researched": researched,
    }
