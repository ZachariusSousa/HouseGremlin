"""Researcher tests — BFS web-research graph growth with mocked LLM + network.

No live model, no real network: web_search/fetch are monkeypatched to canned
results, the LLM is a deterministic fake keyed by subject (the "pretend to be
Gemma" stand-in), and embeddings are hash-derived.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from pc_memory.app import research as research_mod
from pc_memory.app.db import connect, init_schema
from pc_memory.app.llm import LLMError


class MockResearchLLM:
    """Fake Gemma: per-subject canned extraction, mirrors complete_json contract.

    `sequence` (if given) is popped one response per call — for testing
    multi-source subjects where each URL gets its own answer.
    """

    def __init__(self, table: dict[str, dict] | None = None, sequence: list[dict] | None = None):
        self.table = table or {}
        self.sequence = list(sequence) if sequence is not None else []
        self.calls: list[str] = []  # subjects asked about, in order

    def complete_json(self, prompt: str, *, system: str = "", max_tokens: int = 1200, validate=None):
        m = re.search(r"about (.+?) from the web page text", prompt)
        subject = m.group(1).strip("'\"") if m else ""
        self.calls.append(subject)
        if self.sequence:
            data = json.loads(json.dumps(self.sequence.pop(0)))
        else:
            # Case-insensitive table lookup: a real LLM answers regardless of casing.
            entry = next(
                (v for k, v in self.table.items() if k.lower() == subject.lower()),
                {"triples": [], "facts": [], "related": []},
            )
            data = json.loads(json.dumps(entry))
        try:
            if validate is not None:
                validate(data)
            return data
        except Exception as exc:  # invalid -> same retry-then-fail contract as ChatClient
            raise LLMError(f"invalid research output: {exc}") from exc

    def chat(self, prompt: str, **kwargs):
        return json.dumps({})


class MockEmbed:
    def __init__(self, dim: int = 16):
        self.dim = dim
        self.calls = 0

    def probe(self, timeout: float = 5.0) -> bool:
        return True

    def embed(self, texts: list[str], **kwargs):
        self.calls += 1
        out = []
        for text in texts:
            digest = hashlib.sha256(str(text).encode("utf-8")).digest()
            out.append([b / 255.0 for b in digest[: self.dim]])
        return out


PAGES = {
    "https://a.example/mkw": "Mario Kart Wii page text: a kart racing game for the Wii.",
    "https://b.example/mkw": "Another Mario Kart Wii source with more detail.",
    "https://a.example/nintendo": "Nintendo page text.",
    "https://a.example/wii": "Wii console page text.",
}

TABLE = {
    "mario kart wii": {
        "triples": [
            {"subject": "Mario Kart Wii", "predicate": "developed_by", "object": "Nintendo EAD"},
            {"subject": "Mario Kart Wii", "predicate": "platform", "object": "Wii"},
        ],
        "facts": ["Mario Kart Wii was released in 2008."],
        "related": ["Nintendo", "Wii"],
    },
    "nintendo": {
        "triples": [
            {"subject": "Nintendo", "predicate": "founded_in", "object": "1889"},
        ],
        "facts": [],
        "related": ["Wii"],
    },
    "Wii": {
        "triples": [
            {"subject": "Wii", "predicate": "manufacturer", "object": "Nintendo"},
        ],
        "facts": [],
        "related": [],
    },
}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    conn = connect(tmp_path / "mem.db")
    init_schema(conn)
    llm = MockResearchLLM(TABLE)
    embed = MockEmbed()

    def fake_search(query: str, max_results: int = 5) -> list[str]:
        if query.lower().startswith("mario kart wii"):
            return ["https://a.example/mkw", "https://b.example/mkw"]
        if query.lower().startswith("nintendo"):
            return ["https://a.example/nintendo"]
        if query.lower().startswith("wii"):
            return ["https://a.example/wii"]
        return []

    def fake_fetch(c, url: str, **kwargs):
        return PAGES.get(url)

    monkeypatch.setattr(research_mod, "web_search", fake_search)
    monkeypatch.setattr(research_mod, "fetch_url", fake_fetch)
    yield SimpleNamespace(conn=conn, llm=llm, embed=embed, tmp_path=tmp_path)
    conn.close()


def _facts(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT text FROM nodes WHERE kind='fact' ORDER BY id")]


def test_research_subject_stores_facts_triples_provenance(env):
    summary = research_mod.research_subject(
        "mario kart wii", env.conn, llm=env.llm, embed=env.embed, max_sources=3
    )
    assert summary["facts_added"] == 3
    assert summary["sources"] == 2
    assert "Wii" in summary["related"] and "Nintendo" in summary["related"]

    texts = _facts(env.conn)
    assert any("Nintendo EAD" in t for t in texts)
    # Web provenance points at the real source URLs.
    refs = [r[0] for r in env.conn.execute(
        "SELECT source_ref FROM provenance WHERE source_kind='web'"
    )]
    assert set(refs) <= {"https://a.example/mkw", "https://b.example/mkw"}
    # Embeddings were requested for every stored claim.
    assert env.embed.calls >= 1


def test_research_subject_dedupes_across_sources(env):
    # Both sources yield the SAME claim -> merged, not duplicated.
    dup_table = {
        "mario kart wii": {
            "triples": [],
            "facts": ["Mario Kart Wii was released in 2008."],
            "related": [],
        }
    }
    llm = MockResearchLLM(dup_table)
    summary = research_mod.research_subject("mario kart wii", env.conn, llm=llm, embed=None)
    assert summary["facts_added"] == 1
    assert len(_facts(env.conn)) == 1


def test_research_subject_skips_bad_source_and_continues(env):
    # Source A returns invalid extraction (empty subject -> validator rejects),
    # source B is fine. The good source must still be stored.
    bad = {"triples": [{"subject": "", "predicate": "x", "object": "y"}], "facts": [], "related": []}
    good = {"triples": [], "facts": ["Mario Kart Wii was released in 2008."], "related": []}
    llm = MockResearchLLM(sequence=[bad, good])
    summary = research_mod.research_subject("mario kart wii", env.conn, llm=llm, embed=None)
    assert summary["facts_added"] == 1
    assert len(_facts(env.conn)) == 1


def test_research_topic_bfs_grows_frontier_and_persists_state(env):
    state_path = env.tmp_path / "research_state.json"
    summary = research_mod.research_topic(
        "mario kart wii", env.conn, llm=env.llm, embed=None,
        state_path=state_path, max_depth=1, breadth_per_level=1,
    )
    # breadth=1: seed at depth 0, only "Nintendo" researched at depth 1.
    assert summary["subjects_researched"] == 2
    state = research_mod.load_research_state(state_path)
    assert state.visited == {"mario kart wii": 0, "nintendo": 1}
    # "Wii" was discovered but not yet researched -> sits in the frontier.
    assert state.frontier == ["Wii"]


def test_research_topic_resumes_without_revisiting(env):
    state_path = env.tmp_path / "research_state.json"
    first = research_mod.research_topic(
        "mario kart wii", env.conn, llm=env.llm, embed=None,
        state_path=state_path, max_depth=1, breadth_per_level=1,
    )
    calls_before = len(env.llm.calls)
    assert first["subjects_researched"] == 2
    second = research_mod.research_topic(
        "mario kart wii", env.conn, llm=env.llm, embed=None,
        state_path=state_path, max_depth=1, breadth_per_level=1,
    )
    # Second run only touches the remaining frontier (Wii), never revisits.
    assert second["subjects_researched"] == 1
    assert env.llm.calls[calls_before:] == ["Wii"]


def test_research_topic_no_network_results_is_harmless(env, monkeypatch):
    monkeypatch.setattr(research_mod, "web_search", lambda q, max_results=5: [])
    state_path = env.tmp_path / "research_state.json"
    summary = research_mod.research_topic(
        "mario kart wii", env.conn, llm=env.llm, embed=None,
        state_path=state_path, max_depth=1,
    )
    assert summary["subjects_researched"] == 1
    assert summary["totals"]["facts_added"] == 0
    assert len(_facts(env.conn)) == 0
