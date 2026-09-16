# Plan: pc_memory — knowledge graph memory with routed retrieval

## Context

Robit needs durable semantic memory that autonomous agents can grow over time. The user
wants `pc_memory/` (currently empty) to be a **knowledge graph** where agents slowly add
facts derived from scouring the internet or asking the owner questions — e.g.
"sky is blue" → *because* → "Rayleigh scattering by molecules in the atmosphere".

Retrieval goal: given a question, find an **entrance node**, then use a **routing model**
to scour the graph for the nodes most likely to answer it; all discovered nodes are added
to context. Endgame is a learned routing model; we start with LLM-as-router and log every
retrieval trace as its training data.

This intentionally pivots DESIGN.md's "RAG before GraphRAG" guidance for this component —
graph-shaped data + LLM routing is the point here. `DESIGN.md` will be updated so the two
don't contradict (pc_memory becomes the substrate Gate 4 semantic memory builds on).

### Does it make sense?

Yes — with three caveats the design must handle:

1. **Extraction quality is the bottleneck.** Garbage triples → garbage graph. Every fact
   carries provenance; equivalent facts merge; contradictions are flagged, not silently
   overwritten.
2. **LLM routing is slow and token-hungry on a local 4B model.** Budget hops (≤3) and
   candidate beam (≤6), early-stop when the router says "enough". Ingestion stays
   background/slow; retrieval must stay bounded (~seconds, not minutes).
3. **Entrance finding still needs hybrid search** (FTS5 + vectors). The router improves on
   a good seed set; it doesn't replace lexical/semantic entry points.

## Approach

### Graph model

One graph, two node kinds, typed edges:

- **`nodes`** — `kind ∈ {entity, fact}`. Entities are concepts/people/places ("sky",
  "Rayleigh scattering"). Facts are claims with a human-readable sentence, confidence,
  and ≥1 provenance row ("The sky appears blue because Rayleigh scattering preferentially
  scatters short wavelengths"). Every node has `text` (retrievable) and an embedding BLOB.
- **`edges`** — typed relationships between nodes: triple predicates between entity nodes
  (`subject–predicate–object`, matching DESIGN.md's `MemoryFact` contract), plus
  explanation-chain edges between/around fact nodes (`explains`, `caused_by`, `part_of`,
  `related_to`). Unique `(src, dst, type)`.
- **`provenance`** — source kind (`web` | `owner_qa` | `manual`), source ref (URL or the
  owner's question), extracted snippet, confidence, timestamp. Forgetting cascades here.
- **`retrieval_traces`** — question, seed ids, per-hop decisions (candidates, chosen,
  reason), visited ids, rendered context, optional feedback. This is the future router's
  training set; schema is designed for it from day one.

### Storage

SQLite in WAL mode (single file under `pc_memory/data/`), FTS5 external-content table over
node text, embeddings stored as BLOBs next to nodes. The vector index is fully rebuildable
from source rows (DESIGN.md requirement). No extra services.

Embeddings: call the same llama-server's OpenAI-compatible `/v1/embeddings`. **Risk:** a
4B chat GGUF may not expose an embedding head — retrieval degrades gracefully to FTS5 +
graph structure when the endpoint is unavailable (flagged in `GET /health`).

### Ingestion pipeline (slow, background)

`text | url → fetch/chunk → LLM extracts candidate triples + fact sentences + edge types
(strict JSON, pydantic-validated; invalid output logged and rejected per DESIGN.md) →
dedup/merge → store with provenance`.

- Dedup: exact-normalized entity match; for facts, top-k embedding-similar (cosine ≥ ~0.9)
  or FTS5-similar existing facts go to the LLM with a verdict prompt:
  `duplicate | related | contradiction | new`. Duplicate → merge (keep oldest id, raise
  confidence, add provenance). Contradiction → link `contradicted_by`, lower the older
  fact's confidence. Never silent overwrite.
- Web source: DuckDuckGo via `ddgs` (no API key) → top-N results → fetch pages with
  httpx + BeautifulSoup text extraction. Rate-limited and slow by design; agents call it
  from background loops.

### Routed retrieval (foreground, bounded)

1. Question → FTS5 lexical scores + embedding cosine (when available) → **RRF fusion** →
   top-k seed nodes (the "entrance").
2. Router loop, `max_hops=3`, beam ≤6: candidates = typed edge neighbors of the visited
   set ∪ embedding-similar unvisited nodes (escapes weakly connected regions). The LLM
   gets question + visited summaries + candidate list and returns strict JSON: chosen
   node ids with one-line reasons, or `stop`. Chosen nodes join the visited set.
3. Context assembly: visited nodes ordered by hop/relevance → compact text blocks
   (node text + edge types to neighbors) returned as both structured JSON and rendered
   context string ready to append to an agent's prompt.
4. Every run writes a `retrieval_traces` row.

### Learned routing model (endgame, Phase 4)

Replay logged traces to train/evaluate a small router (e.g. node-pair ranker or tiny
transformer over question+path). A/B against the LLM router on a curated question set;
adopt only if it wins accuracy at lower latency. Phases 1–3 just need the trace schema to
be training-ready.

## Files to create (new component, mirrors pc_brain / pc_tracking layout)

```text
pc_memory/
  README.md               # quickstart, endpoints, env vars
  requirements.txt        # fastapi, uvicorn[standard], httpx, pydantic, python-dotenv,
                          # numpy, ddgs, beautifulsoup4, pytest (versions pinned like pc_brain)
  app/
    __init__.py
    config.py             # MEMORY_* env settings (frozen dataclass, mirrors pc_brain pattern)
    db.py                 # SQLite WAL connection + schema init/migration
    store.py              # node/edge/provenance upserts, dedup/merge, forget cascade
    embed.py              # llama-server /v1/embeddings client + graceful unavailability
    llm.py                # OpenAI-compatible chat client (pattern from pc_brain/app/llm.py)
    extract.py            # triple/fact extraction prompt + pydantic validation
    search.py             # ddgs DuckDuckGo search + page fetch/text extraction
    retrieve.py           # RRF seed fusion, hop router, context assembly, trace logging
    main.py               # FastAPI app + endpoints
    cli.py                # argparse CLI: retrieve, add-fact, ingest-text, ingest-url,
                          # inspect, forget, stats, health
  tests/                  # per-module unit tests (pytest)
```

Endpoints (service on **port 8092**; 8080=brain, 8081=llama, 8091=tracking taken):

- `POST /facts` — add one fact/triple directly (manual or agent-supplied)
- `POST /ingest/text`, `POST /ingest/url` — run extraction pipeline on raw text / URL
- `POST /retrieve` — `{question, max_hops?, beam?}` → `{context, nodes[], trace_id}`
- `GET /nodes?q=`, `GET /nodes/{id}`, `DELETE /nodes/{id}` (forget, cascades edges+provenance)
- `GET /traces/{id}`, `GET /stats`, `GET /health`

## Reuse (existing code to mirror or lift patterns from)

- `pc_brain/app/config.py` — frozen-dataclass Settings + `_bool_env/_int_env/_float_env`
  helpers + dotenv; copy the pattern with `MEMORY_*` names.
- `pc_brain/app/llm.py` — `OpenAICompatibleChatClient` (llama-server, strict-JSON prompt
  style, `think: false`); adapt for extraction/routing prompts.
- `pc_brain/requirements.txt` + `pytest.ini` + per-component venv convention
  (`pc_tracking/.venv`) — same packaging shape.
- DESIGN.md `MemoryFact` contract (subject, predicate, object, source event, confidence,
  timestamps) — adopted as the triple/fact row semantics.
- `docs/architecture.md` service conventions (127.0.0.1 binding, health endpoints).

Also: root `pytest.ini` currently has `pythonpath = pc_brain` only — extend it to
`pythonpath = pc_brain pc_memory` so tests run from the repo root.

## Steps

### Phase 0 — Scaffold

- [ ] Create `pc_memory/` package: requirements.txt, README stub, app skeleton,
      config.py (port 8092, DB path under `pc_memory/data/`, llama-server URL/model envs),
      db.py with schema init (nodes, edges, provenance, retrieval_traces, FTS5 table).

### Phase 1 — Core store

- [ ] store.py: upsert node/entity/fact, typed edge insert, provenance attach.
- [ ] Dedup/merge: normalized entity match; similarity + LLM verdict (duplicate/related/
      contradiction/new); confidence update rules.
- [ ] Forget cascade (node → edges → provenance), inspect/stats queries.
- [ ] Tests: schema init, dedup merge, contradiction flagging, forget cascade.

### Phase 2 — Ingestion pipeline

- [ ] embed.py + llm.py clients (llama-server `/v1/embeddings`, `/chat/completions`).
- [ ] extract.py: strict-JSON extraction prompt; pydantic models; reject-and-log invalid
      output.
- [ ] search.py: ddgs query → top-N URLs → httpx fetch → BeautifulSoup text.
- [ ] `POST /facts`, `POST /ingest/text`, `POST /ingest/url` + CLI equivalents.
- [ ] Tests with a mocked LLM (deterministic JSON) and recorded HTML fixture.

### Phase 3 — Routed retrieval

- [ ] retrieve.py: FTS5 + cosine seeds, RRF fusion; hop router (max_hops=3, beam ≤6,
      strict-JSON choices or stop); context assembly; trace logging.
- [ ] `POST /retrieve`, `GET /traces/{id}` + CLI `retrieve`.
- [ ] Seed script: hand-built sky-is-blue causal chain + distractor facts.
- [ ] Tests: seed ranking, hop budget enforcement, early stop, trace completeness,
      FTS5-only fallback when embeddings unavailable.

### Phase 4 — Learned router (endgame, design-for-now)

- [ ] Trace export helper (CSV/JSONL of question → visited path).
- [ ] Curated question set + evaluation harness stub for LLM-router vs learned-router A/B.
      (No training code until traces accumulate.)

### Docs

- [ ] `pc_memory/README.md`: quickstart, endpoints, env vars, agent usage examples.
- [ ] Update `DESIGN.md` memory section: pc_memory is the graph substrate for semantic
      memory; note the deliberate pivot from "RAG before GraphRAG" for this component.

## Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Gemma E4B fumbles strict JSON extraction/routing | Constrained small schemas, temperature ~0.2, one retry, reject-and-log (never partial writes) |
| 4B model lacks embedding head on llama-server | FTS5-only retrieval path is first-class; `/health` reports mode |
| DuckDuckGo rate limits / flaky pages | Slow background ingestion only, per-URL cache table, skip-and-continue on failure |
| Contradictory facts accumulate | Verdict-based handling + `contradicted_by` links + confidence decay; inspect CLI for manual review |
| Routing cost blows up locally | Hard hop/beam budgets, early-stop token, trace reuse for repeated questions (optional cache) |

## Verification

- `pytest pc_memory/tests` green per phase.
- **End-to-end:** start llama-server + service → `ingest-url` a Rayleigh-scattering page →
  `retrieve "why is the sky blue"` returns context containing the causal chain within
  budget; trace shows seeds → hops → visited nodes.
- **Curated set:** ≥10 questions (multi-hop included) answered from seed corpus with all
  key facts present in returned context.
- **Forget test:** delete a node, confirm edges + provenance gone and FTS/vector indexes
  consistent (`rebuild` command).
- Manual: `GET /stats`, `GET /health`, CLI walkthrough in README.
