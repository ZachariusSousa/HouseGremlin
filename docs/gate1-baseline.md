# Gate 1 Baseline

## Automated baseline

- Captured: 2026-07-14
- Runtime: Python 3.11.9
- Pre-change suite: 18 tests passed
- Gate 1 suite after implementation: 33 tests passed
- Gate 1 evaluation pack: `pc_brain/evals/gate1.json`

## Realtime gateway benchmark

- Gateway: `ws://127.0.0.1:8080/v1/realtime`
- Voice: `serena`
- First session turn to first audio: 0.833 seconds
- Warm turn to first audio: 0.337 seconds
- First session total turn: 0.889 seconds
- Warm total turn: 0.493 seconds
- VRAM before turns: 13,779 MB of 16,376 MB
- VRAM after turns: 13,795 MB of 16,376 MB

Both measured turns passed the 1.5-second foreground latency target. The raw
machine-specific result is written to the ignored
`pc_brain/data/gate1-benchmark.json`.

The realtime benchmark writes ignored machine-specific results to
`pc_brain/data/gate1-benchmark.json` so repeated measurements do not dirty the
repository.

Run it after `Scripts/run.bat` reports that the browser is ready:

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_brain\.venv\Scripts\python.exe .\Scripts\benchmark_gate1.py
```

The output records cold and warm text-to-first-audio latency, total turn time,
transcripts, GPU utilization, and VRAM usage. Robot and microphone acceptance
checks remain physical checks because they require the real chassis and room
audio.
