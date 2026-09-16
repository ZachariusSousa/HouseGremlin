"""FastAPI app for pc_memory — the knowledge-graph memory service (port 8092).

Factory pattern: `create_app(service=None)` builds a fully wired app so tests
can inject a temp DB and mocked LLM/embed clients. The module-level `app` is
for uvicorn:

    uvicorn pc_memory.app.main:app
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pc_memory.app.config import Settings, load_settings
from pc_memory.app.db import connect, init_schema
from pc_memory.app.embed import EmbedClient
from pc_memory.app.extract import ExtractionError
from pc_memory.app.ingest import IngestError, ingest_text, ingest_url
from pc_memory.app.llm import ChatClient, LLMError
from pc_memory.app.retrieve import get_trace, retrieve
from pc_memory.app.store import add_fact
from pc_memory.app.store import stats as store_stats

logger = logging.getLogger("pc_memory.main")


@dataclass
class Service:
    """Wired dependencies for the app (DB connection + LLM + embeddings).

    `llm`/`embed` are typed Any on purpose: tests inject deterministic mocks
    with the same surface (`complete_json`, `chat`, `embed`, `probe`).
    """

    settings: Settings
    conn: sqlite3.Connection
    llm: Any = None
    embed: Any = None
    # Serializes DB access: the single connection is shared across FastAPI's
    # thread-pool workers (sqlite3 connections are not safe for concurrent use).
    lock: threading.RLock = field(default_factory=threading.RLock)

    def close(self) -> None:
        self.conn.close()


def build_service(
    settings: Settings | None = None,
    *,
    llm: Any = None,
    embed: Any = None,
    db_path: str | None = None,
) -> Service:
    """Build a Service; defaults wire the real ChatClient/EmbedClient (lazy — no I/O until used)."""
    settings = settings or load_settings()
    conn = connect(db_path or settings.db_path)
    init_schema(conn)
    return Service(
        settings=settings,
        conn=conn,
        llm=llm if llm is not None else ChatClient(settings),
        embed=embed if embed is not None else EmbedClient(settings),
    )


class FactIn(BaseModel):
    text: str = Field(min_length=1)
    confidence: float = 0.7
    source_kind: str = "manual"
    source_ref: str | None = None


class IngestTextIn(BaseModel):
    text: str = Field(min_length=1)
    confidence: float = 0.7
    source_kind: str = "manual"
    source_ref: str | None = None


class IngestUrlIn(BaseModel):
    url: str = Field(min_length=1)
    confidence: float = 0.7


class RetrieveIn(BaseModel):
    question: str = Field(min_length=1)
    max_hops: int | None = Field(default=None, ge=0, le=5)
    beam: int | None = Field(default=None, ge=1, le=20)


def _safe_judge(llm: Any):
    """Store verdict client that degrades to exact-match dedup when the LLM is down."""
    if llm is None:
        return None

    def judge(prompt: str) -> str:
        try:
            return llm.chat(prompt)
        except LLMError as exc:
            logger.warning("LLM verdict unavailable (%s); exact-match dedup only", exc)
            return ""

    return judge


def _embed_one(embed: Any, text: str) -> list[float] | None:
    if embed is None:
        return None
    vectors = embed.embed([text])
    return vectors[0] if vectors else None


def create_app(service: Service | None = None) -> FastAPI:
    owned = service is None
    if service is None:
        service = build_service()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if owned:
            service.close()

    app = FastAPI(title="pc_memory", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    @app.get("/health")
    def health():
        mode = (
            "hybrid"
            if (service.embed is not None and bool(service.embed.probe()))
            else "fts5_only"
        )
        return {"status": "ok", "embedding_mode": mode}

    @app.get("/stats")
    def stats_endpoint():
        with service.lock:
            return store_stats(service.conn)

    @app.post("/facts")
    def facts_endpoint(body: FactIn):
        try:
            with service.lock:
                result = add_fact(
                    service.conn,
                    body.text,
                    confidence=body.confidence,
                    provenance={
                        "source_kind": body.source_kind,
                        "source_ref": body.source_ref,
                        "snippet": body.text[:500],
                    },
                    judge=_safe_judge(service.llm),
                    embedding=_embed_one(service.embed, body.text),
                )
        except LLMError as exc:
            raise HTTPException(503, f"LLM unavailable: {exc}") from exc
        return {"node_id": result.node_id, "verdict": result.verdict}

    @app.post("/ingest/text")
    def ingest_text_endpoint(body: IngestTextIn):
        try:
            with service.lock:
                return ingest_text(
                    service.conn,
                    body.text,
                    llm=service.llm,
                    embed=service.embed,
                    confidence=body.confidence,
                    source_kind=body.source_kind,
                    source_ref=body.source_ref,
                )
        except ExtractionError as exc:
            # All-or-nothing: invalid extraction means nothing was written.
            raise HTTPException(
                422, f"extraction failed, nothing written: {exc}"
            ) from exc
        except IngestError as exc:
            raise HTTPException(400, str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(503, f"LLM unavailable: {exc}") from exc

    @app.post("/ingest/url")
    def ingest_url_endpoint(body: IngestUrlIn):
        try:
            with service.lock:
                return ingest_url(
                    service.conn,
                    body.url,
                    llm=service.llm,
                    embed=service.embed,
                    confidence=body.confidence,
                )
        except ExtractionError as exc:
            raise HTTPException(
                422, f"extraction failed, nothing written: {exc}"
            ) from exc
        except IngestError as exc:
            message = str(exc)
            status = 502 if "fetch" in message else 400
            raise HTTPException(status, message) from exc
        except LLMError as exc:
            raise HTTPException(503, f"LLM unavailable: {exc}") from exc

    @app.post("/retrieve")
    def retrieve_endpoint(body: RetrieveIn):
        try:
            with service.lock:
                return retrieve(
                    service.conn,
                    body.question,
                    llm=service.llm,
                    embed=service.embed,
                    max_hops=body.max_hops,
                    beam=body.beam,
                    budget_chars=service.settings.context_budget_chars,
                )
        except LLMError as exc:
            raise HTTPException(503, f"LLM unavailable: {exc}") from exc

    @app.get("/traces/{trace_id}")
    def trace_endpoint(trace_id: int):
        with service.lock:
            row = get_trace(service.conn, trace_id)
        if row is None:
            raise HTTPException(404, "trace not found")
        return row

    return app


app = create_app()
