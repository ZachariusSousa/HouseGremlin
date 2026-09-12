"""Tests for trace export + eval harness stub (mocked LLM only, no network)."""

from __future__ import annotations

import csv
import json
import re

import pytest
from pc_memory.app.db import connect, init_schema
from pc_memory.app.eval import (
    load_questions,
    run_eval,
    score_context,
)
from pc_memory.app.retrieve import retrieve
from pc_memory.app.seed import seed_sky_chain
from pc_memory.app.traces import export_traces, load_traces


@pytest.fixture()
def conn(tmp_path):
    c = connect(tmp_path / "eval.db")
    init_schema(c)
    seed_sky_chain(c)
    yield c
    c.close()


class MockRouterLLM:
    """Deterministic router: per-hop keyword plan (mirrors test_retrieve).

    Matches keywords against CANDIDATE lines only (`- id=N (via ...): text`),
    never VISITED lines. Stops when the plan says so or the keyword is absent.
    """

    def __init__(self, plan):
        self.plan = list(plan)
        self.calls = 0

    def complete_json(self, prompt, *, system="", max_tokens=1200, validate=None):
        step = self.plan[min(self.calls, len(self.plan) - 1)]
        self.calls += 1
        if step is None:
            return {"stop": True}
        for cid, text in re.findall(r"- id=(\d+) \(via [^)]*\): (.+)", prompt):
            if step.lower() in text.lower():
                return {"choose": [{"id": int(cid), "reason": f"matches '{step}'"}]}
        return {"stop": True}  # keyword not among candidates -> stop, don't loop


def test_score_context_hits_and_misses():
    hits, misses = score_context("The sky is blue via Rayleigh scattering.", ["rayleigh scattering", "potassium-40"])
    assert hits == ["rayleigh scattering"]
    assert misses == ["potassium-40"]


def test_load_questions_curated_set():
    questions = load_questions()
    assert len(questions) >= 10
    for item in questions:
        assert item["id"] and item["question"]
        assert isinstance(item.get("expected_facts"), list) and item["expected_facts"]
    multi_hop = [q for q in questions if len(q["expected_facts"]) >= 3]
    assert multi_hop, "curated set must include multi-hop questions"


def test_load_questions_rejects_bad_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_questions(bad)
    missing = tmp_path / "nope.json"
    with pytest.raises(ValueError):
        load_questions(missing)


def test_export_traces_jsonl_and_csv(conn, tmp_path):
    router = MockRouterLLM(["rayleigh", "wavelength", None])
    retrieve(conn, "Why is the sky blue?", llm=router)
    retrieve(conn, "How many hearts does an octopus have?", llm=None)

    jsonl_path = tmp_path / "traces.jsonl"
    count = export_traces(conn, jsonl_path, fmt="jsonl")
    assert count == 2
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["question"] == "Why is the sky blue?"
    assert first["visited_ids"] and isinstance(first["decisions"], list)

    csv_path = tmp_path / "traces.csv"
    count = export_traces(conn, csv_path, fmt="csv")
    assert count == 2
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    chain_row = next(r for r in rows if r["question"] == "Why is the sky blue?")
    assert chain_row["visited_ids"].count("|") >= 3  # seeds + walked nodes
    assert int(chain_row["hop_count"]) >= 1

    with pytest.raises(ValueError):
        export_traces(conn, tmp_path / "x", fmt="xml")


def test_load_traces_roundtrip(conn):
    retrieve(conn, "Why is the sky blue?", llm=None)
    traces = load_traces(conn)
    assert len(traces) == 1
    assert traces[0]["question"] == "Why is the sky blue?"
    assert isinstance(traces[0]["seed_ids"], list)


def test_run_eval_seed_only_scores_expected_facts(conn):
    questions = [
        {"id": "octopus", "question": "How many hearts does an octopus have?", "expected_facts": ["three hearts"]},
        {"id": "eyes", "question": "What color is the scattered sunlight reaching our eyes?", "expected_facts": ["predominantly blue"]},
    ]
    results = run_eval(conn, questions, llm=None)
    assert [r.score for r in results] == [1.0, 1.0]
    assert all(r.trace_id is not None for r in results)


def test_run_eval_with_mock_router_walks_chain(conn):
    router = MockRouterLLM(["rayleigh", "wavelength", None])
    questions = [
        {
            "id": "full-chain",
            "question": "Why is the sky blue?",
            "expected_facts": ["rayleigh scattering", "shorter wavelengths", "predominantly blue"],
        }
    ]
    results = run_eval(conn, questions, llm=router)
    assert results[0].score == 1.0, results[0].misses


def test_run_eval_learned_router_is_documented_placeholder(conn):
    with pytest.raises(NotImplementedError):
        run_eval(conn, router="learned")
    with pytest.raises(ValueError):
        run_eval(conn, router="bogus")
