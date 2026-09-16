# pc_memory — knowledge-graph memory

SQLite-backed knowledge graph for HouseGremlin's autonomous agents: entity/fact
nodes, typed edges, provenance, FTS5 + optional-embedding hybrid search, and
LLM-routed retrieval with trace logging. The service runs on port **8092**
(ports: 8080=brain, 8081=llama-server, 8091=tracking, 8092=memory).

Phase history and design decisions: `TODO.md`, `Plan.md`. Repo-level rationale:
`DESIGN.md` → "Memory Design" (note the deliberate pivot from "RAG before
GraphRAG" for this component).

## Quickstart

```powershell
# from repo root C:\dev\HouseGremlin
.\pc_memory\.venv\Scripts\python.exe -m pip install -r pc_memory\requirements.txt

# tests (fully mocked LLM; no network, no live model)
.\pc_memory\.venv\Scripts\python.exe -m pytest pc_memory/tests -q

# seed the demo corpus (idempotent sky-is-blue causal chain + distractors)
.\pc_memory\.venv\Scripts\python.exe -m pc_memory.app.cli seed

# start the service on 8092
.\pc_memory\.venv\Scripts\python.exe -m uvicorn pc_memory.app.main:app --port 8092
```

If no embedding endpoint is reachable, the service degrades to FTS5-only mode
and `GET /health` reports `"embedding_mode": "fts5_only"`. If the chat LLM is
down, retrieval degrades to seed-only context with a warning; ingest endpoints
return 503.

## Service (port 8092)

| Endpoint | Body | Returns |
| --- | --- | --- |
| `GET /health` | — | `{"status": "ok", "embedding_mode": "hybrid" \| "fts5_only"}` |
| `GET /stats` | — | node/edge/provenance counts |
| `POST /facts` | `{text, confidence?, source_kind?, source_ref?}` | `{node_id, verdict}` (`duplicate` / `related` / `contradiction` / `new`) |
| `POST /ingest/text` | `{text, confidence?, source_kind?, source_ref?}` | extraction summary (all-or-nothing; 422 on invalid LLM output) |
| `POST /ingest/url` | `{url, confidence?}` | extraction summary (502 on fetch failure) |
| `POST /retrieve` | `{question, max_hops?, beam?}` | `{context, nodes[], trace_id, seed_ids, visited_ids, decisions, ...}` |
| `GET /traces/{id}` | — | full `retrieval_traces` row (seeds, per-hop decisions, rendered context) |

Every `POST /retrieve` writes a `retrieval_traces` row — training data for the
future learned router.

## Environment variables (`MEMORY_*`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEMORY_PORT` | `8092` | service port |
| `MEMORY_DB_PATH` | `pc_memory/data/memory.db` | SQLite path (relative paths resolve under `pc_memory/`) |
| `MEMORY_LLM_BASE_URL` | `http://127.0.0.1:8081/v1` | OpenAI-compatible chat endpoint |
| `MEMORY_LLM_MODEL` | `ggml-org/gemma-4-E4B-it-GGUF:Q4_0` | model name passed to the LLM |
| `MEMORY_LLM_TIMEOUT` | `30.0` | per-request timeout (seconds) |
| `MEMORY_CONTEXT_BUDGET_CHARS` | `6000` | hard rendered-context budget; lowest-relevance nodes are dropped until under budget |
| `MEMORY_MAX_HOPS` | `3` | default hop-router depth |
| `MEMORY_BEAM` | `6` | default candidate beam width |

## Agent usage example

```powershell
# remember a fact (dedup/merge/contradiction judged by the LLM, never silent overwrite)
curl -s -X POST http://127.0.0.1:8092/facts -H 'Content-Type: application/json' `
  -d '{"text": "The greenhouse thermostat is on the north wall.", "source_kind": "agent", "source_ref": "conv-42"}'

# retrieve routed context for a question
curl -s -X POST http://127.0.0.1:8092/retrieve -H 'Content-Type: application/json' `
  -d '{"question": "why is the sky blue"}'
```

The `context` string is pre-rendered, budget-bounded text (node sentences with
edge-type annotations) safe to paste into a prompt. `nodes[]` carries ids,
kinds, and hop depths if you need structured access; `trace_id` links back to
the logged traversal.

## CLI walkthrough

All commands run as `.\pc_memory\.venv\Scripts\python.exe -m pc_memory.app.cli <cmd>`:

| Command | Purpose |
| --- | --- |
| `add-fact TEXT [--confidence F] [--source-kind K] [--source-ref R]` | add one canonical fact node with provenance |
| `ingest-text TEXT [--confidence F]` | LLM-extract triples/facts from raw text (all-or-nothing) |
| `ingest-url URL [--confidence F]` | web search + fetch + extract (per-URL cache, skip-and-continue) |
| `seed` | idempotently load the sky-is-blue demo corpus |
| `retrieve QUESTION [--max-hops N] [--beam N]` | routed retrieval; prints context + trace summary |
| `inspect NODE_ID` | show a node with its edges and provenance |
| `forget NODE_ID` | delete a node and cascade its edges/provenance/FTS rows |
| `stats` | node/edge/provenance counts |
| `health` | health + embedding mode |
| `rebuild` | re-sync the FTS5 index from source rows |
| `export-traces --out PATH [--format jsonl\|csv]` | export `retrieval_traces` for offline analysis/training |

## Retrieval model

1. **Seeds** — question tokens (stopwords filtered) hit the FTS5 external-content
   table; when embeddings are available, cosine-similar nodes are added. Both
   score lists are fused with reciprocal-rank fusion into top-k entrance nodes.
2. **Hop router** — up to `max_hops` (default 3) LLM hops: candidates are typed
   edge neighbors of the visited set plus embedding-similar unvisited nodes
   (beam ≤ 6). The LLM returns strict JSON: chosen candidate ids with one-line
   reasons, or `stop`.
3. **Context assembly** — visited nodes ordered by hop/relevance, rendered as
   compact text blocks, then trimmed from the lowest-relevance end until under
   `MEMORY_CONTEXT_BUDGET_CHARS` (hard budget).

Contradictory facts coexist: a `contradiction` verdict links both fact nodes
with `contradicted_by` edges and keeps both active — age never lowers
confidence.

## Evaluation groundwork

- `evals/questions.json` — 12 curated questions over the seed corpus (single-hop
  and multi-hop).
- `app/eval.py` — harness stub: scores returned context against expected key
  facts for LLM-router vs learned-router A/B. The learned side is a documented
  `NotImplementedError` placeholder until traces accumulate and a model is
  trained (`export-traces` feeds that pipeline).
