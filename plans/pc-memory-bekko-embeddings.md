# pc_memory — Bekko Embedding Integration Plan

Status: **Phases 1–3 DONE (2026-09-15)**. Phases 4–6 pending.

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
3. Background ingestion driver (cron or pc_brain loop) so the graph grows while away.
4. Feedback endpoint `POST /traces/{id}/feedback`; start collecting real traces → Phase 4 learned router.

## Ops notes
- Re-embed command: `./pc_memory/.venv/Scripts/python.exe -m pc_memory.app.cli re-embed`
- Health check: `... cli health` → expect `"embedding_mode": "hybrid"`.
- If 8093 is down, everything still works in FTS5-only mode (by design).
