# pc_memory — knowledge-graph memory (Pi Long Task TODO)

Converted from `pc_memory/Plan.md`. Work through the TODOs in order; do not re-plan this file. Make reasonable implementation decisions yourself; keep it a working MVP.

## Context

HouseGremlin needs durable semantic memory for autonomous agents: a SQLite-backed knowledge graph under `pc_memory/` where agents add facts (entity nodes, fact/claim nodes with provenance, typed edges). Retrieval finds an entrance node via hybrid search (FTS5 + optional embeddings), then an LLM router hops through the graph to assemble bounded context. Every retrieval is logged as training data for a future learned router.

New component mirrors `pc_brain/` / `pc_tracking/` layout: FastAPI service on **port 8092**, SQLite WAL DB under `pc_memory/data/`, OpenAI-compatible LLM client (pattern from `pc_brain/app/llm.py`). Ports: 8080=brain, 8081=llama-server, 8091=tracking, 8092=memory.

## Mandatory corrections (non-negotiable; override Plan.md where they conflict)

1. **Windows PowerShell only — never bash.** All shell steps are PowerShell commands run from repo root `C:\dev\HouseGremlin`.
2. **`pc_memory` must be a proper package with no ambiguous `app` imports.** Create `pc_memory/__init__.py` and `pc_memory/app/__init__.py`; every internal import is absolute `from pc_memory.app.<mod> import ...`. Root `pytest.ini` becomes `pythonpath = . pc_brain` (repo root resolves the `pc_memory` package; do NOT add bare `pc_memory` to pythonpath — that would make `import app` ambiguous with `pc_brain/app`).
3. **Contradictory facts coexist.** On a `contradiction` verdict, link the two fact nodes with `contradicted_by` edges and keep both active. Do NOT reduce a fact's confidence merely because it is older (overrides Plan.md's "lower the older fact's confidence").
4. **Hard rendered-context budget.** Context assembly enforces a hard size budget (`MEMORY_CONTEXT_BUDGET_CHARS`, default 6000) and drops lowest-relevance nodes until under budget. Test this explicitly.
5. **Fact/assertion nodes are the canonical representation of claims.** Every claim lives on a `fact` node (sentence + confidence + ≥1 provenance row). Entity↔entity triple edges link entities to their fact node; edges do not carry claims by themselves.
6. **FTS5 stays synchronized.** The FTS5 external-content table over `nodes.text` must be kept in sync on every node insert/update/delete (SQLite triggers or equivalent app-level writes), plus a `rebuild` CLI command that re-syncs FTS + vector index from source rows.

## Operational rules

- Preserve existing HouseGremlin functionality; this work is additive under `pc_memory/` plus the one-line root `pytest.ini` change.
- Keep implementations simple and MVP-focused; no speculative features.
- Run relevant tests after each phase (`pytest pc_memory/tests -q` from repo root); fix failures before moving on. Never retry the same failing approach more than twice — change tack or record the blocker.
- Commit completed working phases as checkpoints (PowerShell `git add` / `git commit`).
- LLM endpoints: default llama-server at `http://127.0.0.1:8081/v1`. If 8081 is down, try `http://127.0.0.1:8888/v1` (unsloth server) for chat; embeddings may be unavailable → the FTS5-only path must work and `/health` must report the mode. All unit tests use a mocked LLM (deterministic JSON), never a live model.
- If genuinely blocked on a TODO: save current work, keep the repo clean/recoverable, mark that TODO's Status items honestly, continue with later TODOs where possible, and record the blocker + exact next step in `tmp/BLOCKER.md`.

## TODO 1 — Phase 0: scaffold pc_memory package

**Goal:** Create the proper Python package skeleton with config and DB schema so `pytest pc_memory/tests` runs green from the repo root.

**Status:**

- [x] Create `pc_memory/__init__.py`, `pc_memory/app/__init__.py`; every internal import is absolute `from pc_memory.app.<mod> import ...` — no bare `app` imports anywhere under `pc_memory/`.
- [x] `requirements.txt` pinned like `pc_brain/requirements.txt`: fastapi, uvicorn[standard], httpx, pydantic, python-dotenv, numpy, ddgs, beautifulsoup4, pytest.
- [x] `app/config.py`: frozen dataclass Settings with `MEMORY_*` env settings (port 8092, DB path under `pc_memory/data/`, LLM base URL default `http://127.0.0.1:8081/v1`, model name, context budget chars, max hops, beam), dotenv loading; mirror the `pc_brain/app/config.py` pattern.
- [x] `app/db.py`: SQLite WAL connection helper + schema init (nodes, edges, provenance, retrieval_traces, url_cache) + FTS5 external-content table over `nodes.text` with sync triggers on insert/update/delete + `rebuild_fts()` helper.
- [x] Root `pytest.ini` → `pythonpath = . pc_brain`; add `pc_memory/pytest.ini` (`pythonpath = ..`) mirroring `pc_brain/pytest.ini`.
- [x] README stub, `.gitignore` entry for `pc_memory/data/*.db*`, smoke test that imports `pc_memory.app.config` and initializes the schema in a temp DB.

**Verify:**

- [x] From repo root (PowerShell): `pytest pc_memory/tests -q` is green.
- [x] No file under `pc_memory/` contains a bare `from app` / `import app` import.

**Done when:** Package imports cleanly as `pc_memory.app.*`, schema + FTS5 init works in a temp DB, smoke test passes from the repo root.

## TODO 2 — Phase 1: core store with dedup/merge and forget cascade

**Goal:** Node/edge/provenance upserts, LLM-verdict dedup/merge with contradiction coexistence, forget cascade, inspect/stats; tests green.

**Status:**

- [x] `app/store.py`: upsert node (entity/fact), typed edge insert (unique `(src, dst, type)`), provenance attach. Fact nodes are canonical claims (correction 5): every claim is a `fact` node with sentence/confidence/≥1 provenance; triple edges link subject/object entities to the fact node.
- [x] Dedup/merge: exact-normalized entity match; for facts, candidate search via FTS5 (+ embedding cosine when available); LLM verdict prompt returning strict JSON `duplicate | related | contradiction | new`. Duplicate → merge (keep oldest id, raise confidence, add provenance). Contradiction → `contradicted_by` edges between the two fact nodes, both stay active, no age-based confidence change (correction 3). Never silent overwrite.
- [x] Forget cascade: delete node → its edges → its provenance + FTS5 rows; inspect/stats queries.
- [x] Tests: schema init, entity dedup merge, fact duplicate merge, contradiction flagging (both facts survive with `contradicted_by`), forget cascade leaves FTS consistent.

**Verify:**

- [x] From repo root: `pytest pc_memory/tests -q` is green.
- [x] No code path lowers confidence solely because a fact is older.

**Done when:** All Phase 1 store tests pass from the repo root with a mocked LLM verdict client.

## TODO 3 — Phase 2: ingestion pipeline (LLM extraction + web)

**Goal:** LLM/embed clients, strict-JSON extraction, web search/fetch, `POST /facts` + `/ingest/text` + `/ingest/url` endpoints and CLI equivalents; mocked-LLM tests green.

**Status:**

- [x] `app/llm.py`: OpenAI-compatible chat client (pattern from `pc_brain/app/llm.py`), strict-JSON prompt style, `think: false`, temperature ~0.2, one retry on parse failure.
- [x] `app/embed.py`: `/v1/embeddings` client with graceful unavailability (returns None; service degrades to FTS5-only and `/health` reports the mode).
- [x] `app/extract.py`: extraction prompt → pydantic models (triples, fact sentences, edge types); invalid LLM output rejected and logged — never partial writes.
- [x] `app/search.py`: ddgs DuckDuckGo query → top-N URLs → httpx fetch → BeautifulSoup text; per-URL cache table; rate-limited; skip-and-continue on failure.
- [x] `app/main.py` FastAPI app + endpoints: `POST /facts`, `POST /ingest/text`, `POST /ingest/url`, `GET /health` (reports embedding mode), `GET /stats`; `app/cli.py` argparse CLI: `add-fact`, `ingest-text`, `ingest-url`, `inspect`, `stats`, `health`.
- [x] Tests with a mocked LLM (deterministic JSON) and a recorded HTML fixture; no live-model tests.

**Verify:**

- [x] From repo root: `pytest pc_memory/tests -q` is green.
- [x] Live-service smoke (PowerShell): start uvicorn on 8092, `POST /facts` one fact, `GET /stats` shows it, stop the service.

**Done when:** Ingestion endpoints work end-to-end in tests with a mocked LLM and pass the live-service smoke test.

## TODO 4 — Phase 3: routed retrieval with trace logging

**Goal:** FTS5+cosine seeds → RRF fusion → hop router (≤3 hops, beam ≤6) → budgeted context assembly + trace logging; `POST /retrieve` + CLI + seed script; tests green.

**Status:**

- [x] `app/retrieve.py`: seeds = FTS5 lexical scores + embedding cosine (when available) → RRF fusion → top-k entrance nodes. Router loop `max_hops=3`, beam ≤6: candidates = typed edge neighbors of visited set ∪ embedding-similar unvisited nodes; LLM returns strict JSON chosen node ids with one-line reasons or `stop`; early-stop on `stop`. (FTS seeds filter stopwords so function words can't pollute the seed set.)
- [x] Context assembly: visited nodes ordered by hop/relevance → compact text blocks (node text + edge types to neighbors); enforce the hard rendered-context budget (`MEMORY_CONTEXT_BUDGET_CHARS`, default 6000) by dropping lowest-relevance nodes until under budget (correction 4). Return both structured JSON and a rendered context string.
- [x] Every run writes a `retrieval_traces` row: question, seed ids, per-hop decisions (candidates/chosen/reason), visited ids, rendered context, optional feedback — training-ready schema.
- [x] `POST /retrieve` `{question, max_hops?, beam?}` → `{context, nodes[], trace_id}`; `GET /traces/{id}`; CLI `retrieve` (+ `seed`).
- [x] Seed script: hand-built sky-is-blue causal chain (sky → Rayleigh scattering → short-wavelength scatter) + distractor facts — idempotent (`app/seed.py`, CLI `seed`).
- [x] Tests: seed ranking, hop budget enforcement, early stop, trace completeness, FTS5-only fallback when embeddings unavailable, context-budget truncation, seed idempotency (9 tests in `tests/test_retrieve.py`).

**Verify:**

- [x] From repo root: `pytest pc_memory/tests -q` is green (33 passed).
- [x] CLI `retrieve "why is the sky blue"` on the seeded DB returns context containing the causal chain within budget (verified; degrades to seed-only with a stderr warning when the live LLM at 8081 is down).

## TODO 5 — Phase 4: learned-router groundwork (design-for-now)

**Goal:** Trace export, curated question set, evaluation harness stub for future LLM-vs-learned router A/B. No training code until traces accumulate.

**Status:**

- [x] Trace export helper: CSV/JSONL of question → visited path; CLI `export-traces`. (`app/traces.py`: `load_traces` + `export_traces` with full-fidelity JSONL and flat CSV.)
- [x] Curated question set (≥10 questions, multi-hop included) as JSON under `pc_memory/evals/`. (`evals/questions.json`: 12 questions over the sky chain + distractors; `full-chain` requires ≥3 expected facts across hops.)
- [x] Evaluation harness stub that scores returned context against expected key facts for LLM router vs learned router A/B (learned side is a documented placeholder). (`app/eval.py`: `score_context` case-insensitive substring scoring, `run_eval` over the curated set; `router="learned"` raises `NotImplementedError` with a pointer to `retrieval_traces` until a model is trained.)

**Verify:**

- [x] From repo root: `pytest pc_memory/tests -q` is green (41 passed).
- [x] `export-traces` produces a non-empty file from the seeded DB (verified both `jsonl` and `csv` on a scratch DB).

**Done when:** Traces export cleanly, curated set + harness stub are in place, tests pass.

## TODO 6 — Docs, end-to-end verification, final checkpoint

**Goal:** README complete, DESIGN.md updated, full test suite green (including pc_brain regression), e2e run with live LLM if available, repo committed and clean.

**Status:**

- [ ] `pc_memory/README.md`: quickstart, endpoints, env vars, agent usage examples, CLI walkthrough.
- [ ] Update `DESIGN.md` memory section: pc_memory is the graph substrate Gate 4 semantic memory builds on; note the deliberate pivot from "RAG before GraphRAG" for this component.
- [ ] Full suite from repo root: `pytest pc_memory/tests -q` and `pytest pc_brain/tests -q` both green (regression check).
- [ ] E2E: if a live LLM is available (try 8081, else 8888): start the service on 8092, `ingest-url` a Rayleigh-scattering page, `retrieve "why is the sky blue"` returns context containing the causal chain within budget; trace shows seeds → hops → visited. If no live LLM works: run the equivalent flow against the mocked-LLM harness and record that e2e was mock-verified.
- [ ] Curated set check: ≥10 questions answered from the seed corpus with all key facts present in returned context (mocked or live).
- [ ] Forget test: delete a node, confirm edges + provenance gone and FTS/vector indexes consistent after `rebuild`.
- [ ] Manual: `GET /stats`, `GET /health`; commit all completed work as the final checkpoint.

**Verify:**

- [ ] `git status` clean (only intentional artifacts), all tests green, e2e evidence recorded in the task result.

**Done when:** Docs complete, all tests green, e2e evidence recorded, repo committed and recoverable.

## Progress

- [x] TODO 1 — Phase 0: scaffold pc_memory package
- [x] TODO 2 — Phase 1: core store with dedup/merge and forget cascade
- [x] TODO 3 — Phase 2: ingestion pipeline (LLM extraction + web)
- [x] TODO 4 — Phase 3: routed retrieval with trace logging
- [x] TODO 5 — Phase 4: learned-router groundwork (design-for-now)
- [ ] TODO 6 — Docs, end-to-end verification, final checkpoint
