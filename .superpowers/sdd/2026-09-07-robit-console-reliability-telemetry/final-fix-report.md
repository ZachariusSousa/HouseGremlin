# Final fix wave report

Base: `55957c022495c520059f577a0cbed79501cfe8d8`

## Result

All nine findings in `final-fix-brief.md` were fixed test-first without changing the robot's actuation limits, leases, watchdog priority, or public response fields used by existing clients. New response fields are additive.

## Fix and test notes

1. Realtime attempts now have a monotonically increasing owner generation. Health, audio resume, microphone acquisition, worklet setup, socket creation, and callbacks all re-check ownership; stale attempts dispose only their local resources. Deferred-microphone tests cover disconnect, leaving Voice, and replacement-connect races.
2. Policy cancellations now carry a typed `execution_outcome`. The coordinator journals cancellation/rejection, returns the body to stationary, and does not raise a critical fault; untyped dispatch/transport/acknowledgement failures retain the critical path. Tests cover authorization expiry, manual lease suppression, and a true returned failure.
3. `/chat/action` adds `execution_error`, replaces false success prose on failed actuation, and the console retains the prompt in retry-ready state. Tests drive a real aggregate head failure and both explicit/implicit frontend failure contracts.
4. LLM probes require an exact configured model in supported `data`/`models` descriptors (`id`, `name`, or `model`). Reachability success no longer clears a failed inference state. Empty, wrong, fuzzy, correct, and ordering cases are covered.
5. Critical/watchdog faults remain highest priority; otherwise disconnected or unready robots are always offline before age thresholds are considered. Connected and disconnected boundary matrices are covered.
6. Camera fetch, body read, detached decode, and publication are generation/visibility guarded. Frame pixels and identity publish atomically; retired URLs are revoked. Deferred success/error races are covered.
7. Telemetry requests abort after five seconds, retain no-overlap visible-only polling, and recompute sample freshness on every tick: `<=3s` online, `>3s` through `15s` stale, and `>15s` offline. A never-settling fetch and exact fake-time boundaries are covered.
8. PC Brain tracking clients/services and every health, telemetry, status, start/stop/chat boundary sanitize public errors. The standalone tracking service applies the same sanitizer to load, health, unavailable, and inference errors. Tests use credential-bearing URLs, bearer values, and named tokens.
9. Diagnostics now renders the camera row from `health.components.camera.status`; a configured camera with degraded capture health no longer appears ready.

Focused tests were first observed failing at the intended assertions, then passing after each corresponding implementation. Representative RED results included stale microphone continuations creating sockets, cancellation creating a critical fault, failed chat actuation showing READY, wrong/empty model lists passing, disconnected cached samples becoming stale, stale camera decodes publishing, stalled telemetry remaining ONLINE, raw credentials reaching APIs, and Diagnostics showing camera READY.

## Verification

PowerShell setup used for the repository's Python 3.11 environment (the checked-in `.venv` site-packages were reused because its original base interpreter is absent on this host):

```powershell
$Python311 = 'C:\Users\z1sou\AppData\Local\Temp\robit-python-3.11.9-embed-amd64\python.exe'
```

- Complete PC Brain suite:

  ```powershell
  & $Python311 -c "import sys,runpy;sys.path[:0]=[r'C:\Users\z1sou\HouseGremlin\pc_brain\.venv\Lib\site-packages',r'C:\Users\z1sou\HouseGremlin\pc_brain'];sys.argv=['pytest','-q','pc_brain/tests'];runpy.run_module('pytest',run_name='__main__')"
  ```

  Result: `227 passed, 1 warning in 29.95s`.

- Complete tracking sidecar suite:

  ```powershell
  & $Python311 -c "import sys,runpy;sys.path[:0]=[r'C:\Users\z1sou\HouseGremlin\pc_brain\.venv\Lib\site-packages',r'C:\Users\z1sou\HouseGremlin\pc_tracking'];sys.argv=['pytest','-q','pc_tracking/tests'];runpy.run_module('pytest',run_name='__main__')"
  ```

  Result: `5 passed, 1 warning in 0.36s`.

- Frontend behavior harness:

  ```powershell
  node pc_brain/tests/web_control_behavior.cjs web_control/index.html
  ```

  Result: `web control behavior: ok`.

- Extracted inline JavaScript `node --check` test:

  ```powershell
  & $Python311 -c "import sys,runpy;sys.path[:0]=[r'C:\Users\z1sou\HouseGremlin\pc_brain\.venv\Lib\site-packages',r'C:\Users\z1sou\HouseGremlin\pc_brain'];sys.argv=['pytest','-q','pc_brain/tests/test_web_control.py::test_inline_javascript_is_syntactically_valid'];runpy.run_module('pytest',run_name='__main__')"
  ```

  Result: `1 passed, 1 warning in 0.29s`.

- Whitespace/static diff check before commit:

  ```powershell
  git diff --check
  ```

  Result: pass; Git emitted only the repository's existing Windows LF-to-CRLF notices.

- Required committed-range check:

  ```powershell
  git diff --check 55957c0..HEAD
  ```

  Result: pass.

The sole pytest warning is the environment's pre-existing `PytestConfigWarning` for unknown option `asyncio_default_fixture_loop_scope`. A rendered browser/hardware smoke was not run under the final time constraint; deterministic frontend behavior and syntax checks passed, and no live robot commands were issued.
