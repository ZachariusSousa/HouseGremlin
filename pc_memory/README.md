# pc_memory — knowledge-graph memory

SQLite-backed knowledge graph for HouseGremlin's autonomous agents: entity/fact nodes,
typed edges, provenance, FTS5 + optional embedding hybrid search, and LLM-routed
retrieval with trace logging. Service runs on port **8092** (see `TODO.md` / `Plan.md`).

## Quickstart (WIP — full docs land in a later phase)

```powershell
# from repo root C:\dev\HouseGremlin
.\pc_memory\.venv\Scripts\python.exe -m pytest pc_memory/tests -q
```

- DB: SQLite WAL under `pc_memory/data/` (`MEMORY_DB_PATH`).
- LLM: OpenAI-compatible, default `http://127.0.0.1:8081/v1` (`MEMORY_LLM_BASE_URL`, `MEMORY_LLM_MODEL`).
- Env settings: `MEMORY_*` (port 8092, context budget, max hops, beam) — see `app/config.py`.
