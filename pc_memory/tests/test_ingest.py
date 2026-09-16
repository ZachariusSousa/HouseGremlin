"""Phase 2 ingestion tests — endpoints + pipeline with fully mocked LLM/embed.

No live model and no network: extraction is a deterministic mock, embeddings are
hash-derived, and URL downloads are monkeypatched to a recorded HTML fixture.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pc_memory.app import search as search_mod
from pc_memory.app.config import load_settings
from pc_memory.app.db import connect, init_schema
from pc_memory.app.llm import LLMError
from pc_memory.app.main import Service, create_app

FIXTURES = Path(__file__).parent / "fixtures"


class MockExtractorLLM:
    """Deterministic ChatClient stand-in.

    `complete_json` mirrors the real client's retry-then-fail contract and runs
    the same validator extract_facts uses; `chat` answers store verdicts.
    """

    def __init__(self, extraction: dict | None = None):
        self.extraction = (
            extraction if extraction is not None else {"triples": [], "facts": []}
        )
        self.calls = 0

    def complete_json(
        self, prompt: str, *, system: str = "", max_tokens: int = 1200, validate=None
    ):
        self.calls += 1
        last: Exception | None = None
        for _ in range(2):  # mirror ChatClient.complete_json retry loop
            data = json.loads(json.dumps(self.extraction))
            try:
                if validate is not None:
                    validate(data)
                return data
            except Exception as exc:  # validator or JSON error -> retry, then raise
                last = exc
        raise LLMError(f"invalid extraction output: {last}")

    def chat(self, prompt: str, **kwargs):
        return json.dumps({"verdict": "new"})


class MockEmbed:
    """Deterministic embedding stand-in (hash-derived unit-ish vectors)."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.available = True
        self.calls = 0

    def probe(self, timeout: float = 5.0) -> bool:
        return self.available

    def embed(self, texts: list[str], **kwargs):
        if not self.available:
            return None
        self.calls += 1
        out = []
        for text in texts:
            digest = hashlib.sha256(str(text).encode("utf-8")).digest()
            out.append([b / 255.0 for b in digest[: self.dim]])
        return out


VALID_EXTRACTION = {
    "triples": [
        {
            "subject": "Rayleigh scattering",
            "predicate": "scatters_most",
            "object": "blue light",
        },
    ],
    "facts": ["Sunsets appear red because the atmospheric path is longer."],
}

# First triple valid, second invalid -> validate_extraction must reject the WHOLE batch.
INVALID_EXTRACTION = {
    "triples": [
        {"subject": "sky", "predicate": "appears", "object": "blue"},
        {"subject": "", "predicate": "x", "object": "y"},
    ],
    "facts": [],
}


@pytest.fixture()
def env(tmp_path):
    # db.connect() sets check_same_thread=False: TestClient runs sync endpoints
    # on worker threads, and Service.lock serializes access to the shared conn.
    conn = connect(tmp_path / "mem.db")
    init_schema(conn)
    llm = MockExtractorLLM(VALID_EXTRACTION)
    embed = MockEmbed()
    service = Service(settings=load_settings(), conn=conn, llm=llm, embed=embed)
    client = TestClient(create_app(service=service))
    yield SimpleNamespace(conn=conn, llm=llm, embed=embed, client=client)
    conn.close()


def _node_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]


def _provenance_rows(conn: sqlite3.Connection, node_id: int) -> list:
    return conn.execute(
        "SELECT source_kind, source_ref FROM provenance WHERE node_id = ?", (node_id,)
    ).fetchall()


def test_health_reports_hybrid_mode(env):
    response = env.client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["embedding_mode"] == "hybrid"  # MockEmbed.probe() is True


def test_stats_endpoint_initial(env):
    response = env.client.get("/stats")
    assert response.status_code == 200
    assert response.json()["nodes"] == 0


def test_facts_endpoint_creates_canonical_fact_with_provenance(env):
    response = env.client.post(
        "/facts",
        json={
            "text": "The sky is blue.",
            "source_kind": "manual",
            "source_ref": "unit-test",
        },
    )
    assert response.status_code == 200, response.text
    node_id = response.json()["node_id"]
    assert isinstance(node_id, int)
    rows = _provenance_rows(env.conn, node_id)
    assert len(rows) == 1
    assert rows[0][0] == "manual"
    assert rows[0][1] == "unit-test"


def test_ingest_text_writes_triples_facts_edges_and_fts(env):
    response = env.client.post(
        "/ingest/text", json={"text": "raw paragraph about the sky"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["triples"]) == 1
    triple_id = body["triples"][0]["node_id"]
    assert len(body["facts"]) == 1

    # Triple edges: subject -> fact and object -> fact (edges carry no claims).
    edge_count = env.conn.execute(
        "SELECT COUNT(*) FROM edges WHERE src = ? OR dst = ?", (triple_id, triple_id)
    ).fetchone()[0]
    assert edge_count >= 2

    # FTS5 external-content index stays in sync on insert.
    hit = env.conn.execute(
        "SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH ?", ("rayleigh",)
    ).fetchone()
    assert hit is not None


def test_ingest_text_invalid_extraction_writes_nothing(env):
    env.llm.extraction = INVALID_EXTRACTION
    response = env.client.post("/ingest/text", json={"text": "whatever"})
    assert response.status_code == 422
    # All-or-nothing: not even the first (valid) triple of the batch was written.
    assert _node_count(env.conn) == 0
    assert env.conn.execute("SELECT COUNT(*) FROM provenance").fetchone()[0] == 0


def test_ingest_url_uses_cache_and_web_provenance(env, monkeypatch):
    fixture_html = (FIXTURES / "rayleigh.html").read_text(encoding="utf-8")
    downloads: list[str] = []

    def fake_download(url: str, timeout: float) -> str:
        downloads.append(url)
        return fixture_html

    monkeypatch.setattr(search_mod, "_download", fake_download)

    url = "https://example.com/rayleigh"
    first = env.client.post("/ingest/url", json={"url": url})
    assert first.status_code == 200, first.text
    second = env.client.post("/ingest/url", json={"url": url})
    assert second.status_code == 200

    # Second ingest served from url_cache: exactly one download total.
    assert downloads == [url]

    # Web provenance attached to the extracted nodes.
    row = env.conn.execute(
        "SELECT source_kind, source_ref FROM provenance WHERE source_kind = 'web'"
    ).fetchall()
    assert row and all(ref == url for _kind, ref in row)


def test_ingest_url_fetch_failure_returns_502(env, monkeypatch):
    def boom(url: str, timeout: float) -> str:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(search_mod, "_download", boom)
    response = env.client.post("/ingest/url", json={"url": "https://example.com/down"})
    assert response.status_code == 502
    assert _node_count(env.conn) == 0
