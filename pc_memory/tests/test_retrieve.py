"""Phase 3 retrieval tests — routed traversal, traces, budgets (fully mocked).

No live model and no network: the hop router is a deterministic mock LLM that
answers the strict-JSON prompt, and seeding uses the idempotent sky-is-blue
chain from app.seed.
"""

from __future__ import annotations

import json
import re
import sqlite3

import pytest
from fastapi.testclient import TestClient
from pc_memory.app.config import load_settings
from pc_memory.app.db import connect, init_schema
from pc_memory.app.main import Service, create_app
from pc_memory.app.retrieve import (
    _rrf_fuse,
    assemble_context,
    fts_seed_ids,
    parse_hop_decision,
    retrieve,
)
from pc_memory.app.seed import seed_sky_chain
from pc_memory.app.store import add_fact

QUESTION = "Why is the sky blue?"


class MockRouterLLM:
    """Deterministic router: per-hop keyword plan.

    Plan entry is a substring to match against candidate lines; None stops.
    Picks the first candidate whose text contains the keyword (case-insensitive).
    """

    def __init__(self, plan: list[str | None]):
        self.plan = plan
        self.calls = 0

    def complete_json(
        self, prompt: str, *, system: str = "", max_tokens: int = 1200, validate=None
    ):
        step = self.plan[min(self.calls, len(self.plan) - 1)]
        self.calls += 1
        if step is None:
            return {"stop": True}
        for cid, text in re.findall(r"- id=(\d+) \(via [^)]*\): (.+)", prompt):
            if step.lower() in text.lower():
                return {"choose": [{"id": int(cid), "reason": f"matches '{step}'"}]}
        return {"stop": True}  # keyword not among candidates -> stop, don't loop

    def chat(self, prompt: str, **kwargs):
        return json.dumps({"verdict": "new"})


class MockAlwaysFirstLLM:
    """Router that always picks the first candidate (never stops on its own)."""

    def __init__(self):
        self.calls = 0

    def complete_json(
        self, prompt: str, *, system: str = "", max_tokens: int = 1200, validate=None
    ):
        self.calls += 1
        match = re.search(r"- id=(\d+) \(via [^)]*\)", prompt)
        if not match:
            return {"stop": True}
        return {"choose": [{"id": int(match.group(1)), "reason": "first candidate"}]}

    def chat(self, prompt: str, **kwargs):
        return json.dumps({"verdict": "new"})


@pytest.fixture()
def conn(tmp_path):
    # db.connect() sets check_same_thread=False: TestClient runs sync endpoints
    # on worker threads, and Service.lock serializes access to the shared conn.
    c = connect(tmp_path / "mem.db")
    init_schema(c)
    seed_sky_chain(c)  # no embed -> pure FTS world, fully deterministic
    yield c
    c.close()


def _client(conn: sqlite3.Connection, llm=None, embed=None) -> TestClient:
    service = Service(settings=load_settings(), conn=conn, llm=llm, embed=embed)
    return TestClient(create_app(service=service))


def _node_id_by_text(conn: sqlite3.Connection, text: str) -> int:
    row = conn.execute("SELECT id FROM nodes WHERE text = ?", (text,)).fetchone()
    assert row is not None, f"missing node: {text[:60]}"
    return row["id"]


def test_rrf_fusion_and_parsing_units():
    # Consensus between two rankings wins over a single ranking.
    scores = _rrf_fuse([[1, 2], [2, 3]])
    top = max(scores, key=lambda node_id: scores[node_id])
    assert top == 2

    # Unparseable router reply -> stop (no infinite loop), unknown ids dropped.
    chosen, reasons, stop = parse_hop_decision("not json at all", [5])
    assert (chosen, stop) == ([], True)
    chosen, reasons, stop = parse_hop_decision(
        {"choose": [{"id": 999, "reason": "x"}, {"id": 5, "reason": "ok"}]}, [5]
    )
    assert (chosen, stop) == ([5], False)
    assert reasons[5] == "ok"


def test_fts_seed_ranking_prefers_chain_over_distractors(conn):
    chain_id = _node_id_by_text(
        conn, "The sky appears blue because of Rayleigh scattering."
    )
    seeds = fts_seed_ids(conn, QUESTION)
    assert seeds, "expected FTS seeds for the question"
    assert seeds[0] == chain_id  # bm25: matches why/sky/blue tokens


def test_retrieve_walks_chain_and_writes_trace(conn):
    llm = MockRouterLLM(["rayleigh", "wavelength", None])
    client = _client(conn, llm=llm)

    response = client.post("/retrieve", json={"question": QUESTION})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "Rayleigh" in body["context"]
    assert "wavelength" in body["context"]
    assert isinstance(body["trace_id"], int)

    trace_response = client.get(f"/traces/{body['trace_id']}")
    assert trace_response.status_code == 200
    trace = trace_response.json()
    assert trace["question"] == QUESTION
    assert isinstance(trace["seed_ids"], list) and trace["seed_ids"]
    assert len(trace["decisions"]) >= 2
    assert set(trace["visited_ids"]) - set(trace["seed_ids"]), (
        "router should walk beyond the seeds"
    )

    # The walked nodes are the chain facts (Rayleigh scattering / wavelengths).
    rayleigh_id = _node_id_by_text(
        conn,
        "Rayleigh scattering is the scattering of light by particles much smaller than its wavelength, such as air molecules.",
    )
    assert rayleigh_id in trace["visited_ids"]


def test_hop_budget_enforced(conn):
    llm = MockAlwaysFirstLLM()
    client = _client(conn, llm=llm)

    response = client.post("/retrieve", json={"question": QUESTION, "max_hops": 2})
    assert response.status_code == 200, response.text
    trace = client.get(f"/traces/{response.json()['trace_id']}").json()
    assert len(trace["decisions"]) == 2  # never more than max_hops router calls


def test_router_early_stop(conn):
    llm = MockRouterLLM([None])  # stops at hop 1
    client = _client(conn, llm=llm)

    response = client.post("/retrieve", json={"question": QUESTION})
    assert response.status_code == 200, response.text
    trace = client.get(f"/traces/{response.json()['trace_id']}").json()
    assert len(trace["decisions"]) == 1
    assert trace["decisions"][0]["stop"] is True
    assert trace["visited_ids"] == trace["seed_ids"], "early stop must not add nodes"


def test_trace_completeness_and_404(conn):
    client = _client(conn, llm=MockRouterLLM(["rayleigh", None]))
    response = client.post("/retrieve", json={"question": QUESTION})
    assert response.status_code == 200, response.text
    trace_id = response.json()["trace_id"]

    trace = client.get(f"/traces/{trace_id}").json()
    for key in ("question", "seed_ids", "decisions", "visited_ids", "context"):
        assert key in trace, f"missing trace field {key}"
    assert isinstance(trace["context"], str) and trace["context"]
    decision = trace["decisions"][0]
    for key in ("hop", "candidates", "chosen", "reasons", "stop"):
        assert key in decision, f"missing decision field {key}"

    assert client.get("/traces/99999").status_code == 404


def test_fts_only_fallback_when_embeddings_unavailable(conn):
    # No embed on the service AND no embedding BLOBs in the DB: retrieval must
    # still work end-to-end via FTS5 seeds alone, and /health says so.
    client = _client(conn, llm=MockRouterLLM(["rayleigh", None]), embed=None)
    assert client.get("/health").json()["embedding_mode"] == "fts5_only"

    response = client.post("/retrieve", json={"question": QUESTION})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "Rayleigh" in body["context"]
    assert isinstance(body["trace_id"], int)


def test_context_budget_truncation(conn):
    # Unit level: a hard char budget drops the lowest-relevance tail first.
    ids = [
        add_fact(conn, f"Fact number {i} about the sky and light scattering.").node_id
        for i in range(10)
    ]
    context, blocks = assemble_context(conn, ids, budget_chars=200)
    assert context and len(context) <= 200
    assert len(blocks) < 10
    assert blocks[0]["id"] == ids[0], "highest-relevance node must survive"

    # Endpoint level: retrieve() honors the same budget.
    result = retrieve(conn, QUESTION, llm=None, budget_chars=200)
    assert result["context"] and len(result["context"]) <= 200
    assert len(result["nodes"]) < 10


def test_seed_is_idempotent(conn):
    before = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edges_before = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    seed_sky_chain(conn)  # second run: everything merges, nothing duplicates
    after = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edges_after = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    assert after == before
    assert edges_after == edges_before
