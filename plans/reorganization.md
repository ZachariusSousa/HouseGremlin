# HouseGremlin Reorganization Plan (2026-09-18)

Goal: one coherent open-source-ready monorepo. Three hard package boundaries,
no god-files, every package runnable in isolation with zero hardware.

**Context rules for the working session(s):**
- This file is self-contained. Do NOT re-survey the repo; trust it until a step fails.
- One phase = one commit. Never start a phase without its test baseline green.
- Baselines (verified 2026-09-18): pc_brain **227 tests**, pc_tracking **5**,
  pc_memory **57**. Venvs: `pc_brain/.venv`, `pc_tracking/.venv`,
  `pc_memory/.venv` (all gitignored). llama.cpp at `C:\Tools\llama.cpp`.
- Windows + Git Bash. `du` on the repo root TIMES OUT (5 GB+ of venvs) — never use it;
  use `git ls-files` / targeted `ls`.
- No LLM/VRAM needed for any phase. Robot may be off.

## Phase 0 — Baseline snapshot (10 min, no commit)
Run all three suites, record pass counts:
```
cd C:/dev/HouseGremlin
./pc_brain/.venv/Scripts/python.exe -m pytest pc_brain/tests -q        # expect 227
./pc_tracking/.venv/Scripts/python.exe -m pytest pc_tracking/tests -q  # expect 5
./pc_memory/.venv/Scripts/python.exe -m pytest pc_memory/tests -q      # expect 57
```
Also verify live smoke (robot optional): `python -m pc_brain.app.main` imports cleanly.

## Phase 1 — Sweep the junk (commit: "chore: remove dead files and tool state")
Delete (all verified zero-reference, see tmp/cleanup-audit.md):
- `.superpowers/` → untrack (`git rm -r --cached .superpowers`), keep on disk, gitignore it.
  Exception: move `.superpowers/sdd/2026-09-07-robit-console-reliability-telemetry/final-fix-report.md`
  to `docs/superpowers/reports/` first if user wants the record.
- `pc_tracking/.pytest_cache/` → untrack, gitignore `*/.pytest_cache/`.
- `pc_brain/app/cuda_paths.py` (29 lines, zero callers).
- `Scripts/patch_speech_to_speech_timeout.py` (one-shot, already applied).
- `pc_brain/streaming-spike*.wav` (3 files, spike artifacts).
- `cad/Robit(50)-Maindesign.stl` — keep only the README-referenced
  `pc_brain/cad/Maindesign.stl` (verify which is newer via git log; if cad/ one is newer, swap refs instead).
- Orphaned Scripts → delete: `benchmark_tracking.py`, `benchmark_vision.py`,
  `collect_vision_corpus.ps1`, `resolve_robot_host.py`. Keep documented ones.
- YuNet plan: **ASK USER** keep-or-delete (0/44 tasks, zero code refs) — default delete.
Gitignore additions: `.superpowers/`, `*/.pytest_cache/`.
Gate: all three suites unchanged.

## Phase 2 — Naming decision (RESOLVED: robot = **Robit**, repo = HouseGremlin; YuNet plan DELETE — user confirmed 2026-09-18)
Robot is called Robit / robot / HouseGremlin interchangeably. Pick ONE canonical name
(recommendation: keep **Robit** for the robot, HouseGremlin for the repo/program).
Grep-replace across code, env var names stay stable where already exported
(`ROBIT_*` stays if we pick Robit — that's the point of picking it first).
Update README + docs. Gate: suites unchanged.

## Phase 3 — Directory restructure (2 commits)
Target layout:
```
brain/        (from pc_brain/)
tracking/     (from pc_tracking/)
memory/       (from pc_memory/)
web/          (from web_control/)
firmware/     (unchanged)
cad/          (single STL, moved up from brain/cad if Phase 1 kept the other)
tools/        (from Scripts/, after orphans deleted)
docs/ plans/ models/ (models stays gitignored)
```
Commit A: `git mv` the dirs. Commit B: fix references —
- `pyproject.toml` package names/paths, `pytest.ini` roots, `.env.example`s
- `tools/supervisor.py` (has hardcoded `BRAIN = ROOT / "pc_brain"`, `TRACKING = ...`)
- all `.bat`/`.ps1` in tools/, README, docs, plans
- pc_memory has NO pyproject.toml (requirements.txt only) — add one for consistency
Gate: all three suites from NEW paths + supervisor dry-run.

## Phase 4 — Split the god-file (2–3 commits, highest risk)
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
- `sanitization.py` is byte-identical in brain/ and tracking/: inline the ~46 lines into
  each (do NOT build a shared package for this).
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
