# pc_memory — Bekko Embedding Integration Plan

Status: **Phases 1–4 DONE (2026-09-15)**. Phase 4b (telemetry ingestion driver) + Phase 4c (autonomous web researcher) DONE (2026-09-17). Phases 5, 6.1/6.2/6.4 pending.

## Model choice

**`hotchpotch/bekko-embedding-v1-a25m-GGUF` → `bekko-embedding-v1-a25m-Q8_0.gguf` (137 MB)**
- 24.9M params, 384-dim output, mean pooling + L2 normalize, 8k context.
- ~57.5 MMTEB retrieval score — beats BGE-M3 with ~12x fewer params; a25 is the
  right size (a8m saves nothing meaningful at this scale, Q8_0 keeps accuracy near F16).
- Served via llama-server `--embedding --pooling mean --embd-normalize 2` on **127.0.0.1:8093**.
- Model file: `C:\dev\HouseGremlin\models\bekko-embedding-v1-a25m-Q8_0.gguf` (gitignored).

## Phase 1 — Serve the model ✅
- Installed llama.cpp b10991 to `C:\Tools\llama.cpp` (where supervisor already looks).
- Server flags: `-m models/bekko-embedding-v1-a25m-Q8_0.gguf --host 127.0.0.1 --port 8093
  --embedding --pooling mean --embd-normalize 2 --ctx-size 8192`
- Verified: `/v1/models` lists it; embeddings are 384-dim, L2 norm = 1.0.

## Phase 2 — Point pc_memory at it ✅
- `config.py`: new `embed_base_url` / `embed_model` settings
  (`MEMORY_EMBED_BASE_URL`, `MEMORY_EMBED_MODEL`; default `http://127.0.0.1:8093/v1`).
  Chat LLM stays on 8081 — embeddings and chat are separate endpoints now.
- `embed.py`: `EmbedClient` uses the dedicated endpoint + model name.

## Phase 3 — Re-embed + verify ✅
- New CLI command: `python -m pc_memory.app.cli re-embed` (store.re_embed batches
  32 nodes, writes float32 blobs, reports failures). Run it after ANY embedding
  backend change — vectors across models are not comparable.
- Real DB re-embedded; `/health` now reports `embedding_mode: hybrid` (was fts5_only).
- Eval harness (keyword router, fresh sky-chain seed): **12/12 questions pass**,
  including `octopus-probe` which failed under FTS-only. Full suite: 41 passed.

## Phase 4 — Supervisor integration ✅
- `Scripts/supervisor.py`: new `embedding-server` Service on 8093, marked `optional=True`:
  - skips cleanly if the GGUF is missing (warn, no crash),
  - never counts against the required-sidecar watch loop (pc_memory degrades to
    FTS5-only when it's down — same as before this change).
- Next boot of `run.bat` brings embeddings up automatically with the robot stack.

## Phase 5 — Retrieval quality (next)
1. Porter stemming on the FTS5 table (schema + rebuild) — cheap win for inflection misses.
2. Re-run evals with the **LLM router** once Gemma is back up; compare against keyword baseline.
3. Add latency metrics to traces (embed ms, llm ms, graph ms) so router A/B has data.

## Phase 6 — Growth loop (the actual point of the component)
1. Missing node endpoints: `GET /nodes?q=`, `GET /nodes/{id}`, `DELETE /nodes/{id}`.
2. Move LLM calls out of the global DB RLock in `/retrieve` and `/facts`.
3. ✅ **Background ingestion driver (DONE 2026-09-17)** — see Phase 4b below.
4. Feedback endpoint `POST /traces/{id}/feedback`; start collecting real traces → Phase 4 learned router.

## Phase 4b — Ingestion driver: robot telemetry → memory graph ✅ (2026-09-17)
The engine's #1 gap was "no fuel": the graph had one seed node and nothing to feed it.
This closes it with **zero LLM** (VRAM-safe; runs while Gemma is down).

- New module `pc_memory/app/brain_ingest.py`:
  - Reads `pc_brain/data/brain.db` → `brain_events` **read-only** (`file:...?mode=ro`).
  - `convert_event()` deterministically turns each structured event into a stable
    claim sentence (faults, action lifecycles, state transitions, eye heartbeats).
    No timestamps in the claim, so repeated occurrences merge into one fact node.
  - Stores via existing `store.add_fact` with provenance
    `source_kind='manual'`, `source_ref='brain_event:<event_id>'` (no schema change).
  - **Per-event idempotency**: sequence watermark is the primary gate; a
    `already_ingested()` check on `source_ref` means a crash/state-reset can never
    double-count an occurrence. Repeated telemetry accumulates one provenance row
    per event → "happened N times" is queryable.
  - `ingest_brain()` (one pass) + `watch_brain()` (poll loop, survives missing/locked DB).
- New CLI: `ingest-brain` (one-shot, `--dry-run`, `--no-embed`) and `watch-brain`
  (`--poll-seconds`). State file lives next to the memory DB, not CWD-relative.
- Verified live: 20 real brain events → 10 unique facts (repeats merged), each event
  exactly once; retrieval of "why did the robot fault?" surfaces the real fault chain.
  Full suite: **51 passed** (41 original + 10 new in `tests/test_brain_ingest.py`).
- Ops note: clean up a double-counted store with one query — keep `MIN(id)` per
  `source_ref`, delete the rest (done once during development).

## Phase 4c — Autonomous web researcher: subject → knowledge graph ✅ (2026-09-17)
The growth loop for *general* subjects (not just robot telemetry): hand it a seed
like "mario kart wii" and, while you're away, it searches the web, extracts facts
via LLM, discovers related subjects, and grows the graph node-by-node — a personal
Wikipedia. Retrieval stays LLM-light: Bekko routes to fact nodes, small LLM synthesizes.

- New module `pc_memory/app/research.py`:
  - **Source-agnostic** (no Wikipedia lock-in): ddgs `web_search` → fetch top N
    sources via existing `search.fetch_url`. Bad/403 sources are skipped, not fatal.
  - **Always LLM, no fallback**: Gemma reads each page and returns one JSON —
    facts + typed triples + related subjects (the frontier). Deterministic code only
    does search/fetch plumbing; extraction and relationship discovery are the LLM's.
  - `research_subject()` stores via existing `store.add_fact`/`add_triple` with
    provenance `source_kind='web'` + source URL, Bekko embeds, dedup/merge as usual.
  - **BFS growth**: related subjects become the next level; state (visited + frontier)
    persists to `research_state.json` next to the memory DB, so it resumes exactly
    where it stopped after a reboot/kill. Case-insensitive visited/frontier keys.
- New CLI: `research <subject>` (one-shot burst) and `watch-research` (drain the
  frontier until empty — the "while you're away" mode). Flags: `--max-depth`,
  `--breadth`, `--max-sources`, `--reset`, `--no-embed`.
- Verified: **6 new tests** in `tests/test_research.py` (all LLM-mocked, VRAM-safe) —
  BFS growth, frontier persistence/resume, cross-source dedup, invalid-extraction
  skip, state reset. Live end-to-end with a fake-Gemma HTTP server on 8081 (real
  search + real fetches + real Bekko embeds into a throwaway DB): researched
  mario kart wii → Nintendo → Wii → Mario Kart; `retrieve "who made mario kart wii"`
  surfaced the developed_by/published_by facts. Full suite: **57 passed**.
- Ops note: to run for real, start Gemma on 8081 + Bekko on 8093, then
  `python -m pc_memory.app.cli watch-research "mario kart wii"`. Wikipedia 403s the
  fetcher — handled by skipping that source; ddgs ranks others (MarioWiki, etc.).

## Ops notes
- Re-embed command: `./pc_memory/.venv/Scripts/python.exe -m pc_memory.app.cli re-embed`
- Health check: `... cli health` → expect `"embedding_mode": "hybrid"`.
- If 8093 is down, everything still works in FTS5-only mode (by design).
