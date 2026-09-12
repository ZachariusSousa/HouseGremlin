"""Evaluation harness stub for LLM-router vs learned-router A/B.

Scores retrieved context against expected key facts (case-insensitive
substring match) so router quality can be compared on a curated question set
(`evals/questions.json`). The *learned* side is a documented placeholder:
until traces accumulate and a model is trained, `router="learned"` raises
`NotImplementedError` — the A/B interface and scoring are already in place.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pc_memory.app.retrieve import retrieve

DEFAULT_QUESTIONS = Path(__file__).resolve().parent.parent / "evals" / "questions.json"


@dataclass
class EvalResult:
    question_id: str
    question: str
    score: float
    hits: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    trace_id: int | None = None
    router: str = "llm"


def load_questions(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the curated question set (list of {id, question, expected_facts})."""
    p = Path(path) if path is not None else DEFAULT_QUESTIONS
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read question set {p}: {exc}") from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"question set {p} is not valid JSON: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise ValueError(f"question set must be a non-empty list: {p}")
    for item in data:
        if "id" not in item or "question" not in item:
            raise ValueError(f"each question needs 'id' and 'question': {item!r}")
    return data


def score_context(context: str, expected_facts: list[str]) -> tuple[list[str], list[str]]:
    """Case-insensitive substring scoring. Returns (hits, misses)."""
    lowered = context.lower()
    hits = [fact for fact in expected_facts if fact.lower() in lowered]
    misses = [fact for fact in expected_facts if fact.lower() not in lowered]
    return hits, misses


def run_eval(
    conn: sqlite3.Connection,
    questions: list[dict[str, Any]] | None = None,
    *,
    llm: Any = None,
    embed: Any = None,
    router: str = "llm",
    max_hops: int | None = None,
    beam: int | None = None,
    budget_chars: int | None = None,
) -> list[EvalResult]:
    """Run every question through the requested router and score the context.

    `router="learned"` is the A/B placeholder for the future learned router;
    it raises until a trained model exists (see module docstring).
    """
    if router == "learned":
        raise NotImplementedError(
            "learned router not available yet: traces are being accumulated in "
            "retrieval_traces for training; wire a trained model here for the A/B"
        )
    if router != "llm":
        raise ValueError(f"unknown router: {router!r}")
    if questions is None:
        questions = load_questions()
    results: list[EvalResult] = []
    for item in questions:
        outcome = retrieve(
            conn,
            item["question"],
            llm=llm,
            embed=embed,
            max_hops=max_hops,
            beam=beam,
            budget_chars=budget_chars,
        )
        hits, misses = score_context(outcome["context"], item.get("expected_facts") or [])
        total = len(hits) + len(misses)
        results.append(
            EvalResult(
                question_id=str(item["id"]),
                question=item["question"],
                score=(len(hits) / total) if total else 1.0,
                hits=hits,
                misses=misses,
                trace_id=outcome.get("trace_id"),
                router=router,
            )
        )
    return results


__all__ = ["DEFAULT_QUESTIONS", "EvalResult", "load_questions", "run_eval", "score_context"]
