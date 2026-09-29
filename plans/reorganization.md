# HouseGremlin Reorganization Plan (2026-09-18) — STATUS: Phases 1,2,3,5,6 DONE; Phase 4 PENDING

Goal: one coherent open-source-ready monorepo. Three hard package boundaries,
no god-files, every package runnable in isolation with zero hardware.

**Context rules for the working session(s):**
- This file is self-contained. Do NOT re-survey the repo; trust it until a step fails.
- One phase = one commit. Never start a phase without its test baseline green.
- Baselines (verified 2026-09-18, still valid after all commits): brain **227**,
  tracking **5**, memory **57** tests. Venvs: `brain/.venv`, `tracking/.venv`,
  `memory/.venv` (all gitignored). llama.cpp at `C:\Tools\llama.cpp`.
- Windows + Git Bash. `du` on the repo root TIMES OUT (5 GB+ of venvs) — never use it;
  use `git ls-files` / targeted `ls`.
- No LLM/VRAM needed for any phase. Robot may be off.

## DONE log (commits, all test-gated at 227/5/57)
- **Phase 1** `1860b3d` — swept junk: .superpowers untracked+gitignored, cuda_paths.py,
  one-shot patch script, YuNet plan (user confirmed delete), spike WAVs, orphan Scripts.
- **Phase 2** `485ad4a` — README Naming section: HouseGremlin = repo/brand,
  Robit = robot. Code already consistently ROBIT_*; no rename needed.
- **Phase 3** `e5a251f` — git mv pc_brain→brain, pc_tracking→tracking, pc_memory→memory
  (87 files, history preserved) + all reference rewrites (42 files: imports, scripts,
  bats, supervisor, pytest/pyright/pyproject, .gitignore, docs).
- **Phase 5** `5f53d1a` — tracking/.env.example + memory/.env.example added; keep-in-sync
  headers on both sanitization.py copies (decision: keep per-service, NOT shared pkg).
- **Phase 6** `f7eb8fe` — README layout table + port map + ASCII diagram; architecture.md
  gains Memory Service section; stale "thin scaffold" copy replaced.

## Phase 4 — Split the god-file (PENDING — its own session, highest risk)
`brain/app/main.py` is 1,693 lines / 80 defs. Extract in this order, tests after EACH:
1. `robot_client.py`: robot_request/robot_get/robot_post/heartbeat/cached_robot_status/
   fetch_robot_camera_frame/camera_urls (~lines 555–735). Pure move + re-export from main
   so imports don't churn.
2. `actions.py`: validate_action_payload/sanitized_action_payload/normalize_llm_action_body/
   execute_robot_action/execute_action_payload/require_model_eye_expression.
3. Shrink main.py to routes + lifespan + DI getters (target < 500 lines).
Keep module-level singletons (`realtime_gateway` etc.) where they are; move only if a step
forces it. Gate after each: 227 green + `import brain.app.main` smoke.

## Phase 5 — Dedup + package hygiene (1 commit)
- `sanitization.py` is byte-identical in brain/ and tracking/. **Decision: keep one copy per
  service** (verified 2026-09-18): brain has 3 consumers (coordinator, main, tracking) so
  inlining would duplicate it 3x; a shared package for 46 lines is over-engineering.
  Add a one-line header comment on each: "Keep in sync with <other>/app/sanitization.py".
- Each package: own README ("run me alone in 5 min"), `.env.example` generated from
  config defaults, pyproject.toml.

## Phase 6 — Docs pass (1 commit)
- Root README: rewrite intro = what talks to what + diagram (ASCII fine), per-package
  run instructions, contributor on-ramp ordered by barrier: memory (no deps) →
  tracking (GPU) → brain (robot+LLM+camera).
- `docs/architecture.md`: rewrite — predates memory/ingestion/researcher.
- Keep plans/ as historical; mark superseded ones.

## Out of scope (separate conversations)
- realtime_gateway.py internals (880 lines, self-contained — split only if asked)
- Any behavior changes, new features, telemetry→memory wiring
- Moving memory to its own repo (revisit once brain/tracking stabilize)

## Session budget
Phase 1+2: one short session. Phase 3: one session (mechanical but wide).
Phase 4: its own session(s), slowest. Phases 5+6: one session.
If a phase stalls >2 attempts, STOP and report — don't improvise structure.
