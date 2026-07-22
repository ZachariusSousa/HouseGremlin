from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pc_brain"))

from app.config import load_settings
from app.vision import LlamaServerVisionAdapter, inspect_frame


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def load_annotations(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Annotations must be a JSON object keyed by frame filename")
    return payload


async def benchmark(frames: list[Path], annotations: dict[str, dict]) -> dict:
    settings = load_settings()
    adapter = LlamaServerVisionAdapter(
        settings.vision_base_url,
        settings.vision_model,
        settings.vision_request_timeout_seconds,
        settings.vision_max_output_tokens,
    )
    await adapter.probe()
    latencies: list[float] = []
    valid = 0
    failures: list[dict] = []
    grounded_hits = 0
    grounded_total = 0
    hallucinations = 0
    forbidden_total = 0
    previous = None
    for frame_path in frames:
        quality = inspect_frame(frame_path.read_bytes(), previous, 180, 0.0, 0.0)
        previous = quality.preview
        started = perf_counter()
        try:
            output = await adapter.infer(quality.image, "Describe the visible scene and important entities.")
            valid += 1
            latencies.append((perf_counter() - started) * 1000.0)
            labels = {entity.label.casefold() for entity in output.entities}
            annotation = annotations.get(frame_path.name, {})
            expected = {str(label).casefold() for label in annotation.get("expected_entities", [])}
            forbidden = {str(label).casefold() for label in annotation.get("forbidden_entities", [])}
            grounded_hits += len(labels & expected)
            grounded_total += len(expected)
            hallucinations += len(labels & forbidden)
            forbidden_total += len(forbidden)
        except Exception as exc:
            failures.append({"frame": frame_path.name, "error": str(exc)})
    total = len(frames)
    p95 = percentile(latencies, 0.95)
    schema_rate = valid / total if total else 0.0
    return {
        "backend": "llama-server",
        "model": settings.vision_model,
        "image_tokens": settings.vision_image_tokens,
        "frames": total,
        "schema_valid": valid,
        "schema_valid_rate": schema_rate,
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": p95,
        "grounded_entity_recall": grounded_hits / grounded_total if grounded_total else None,
        "hallucination_rate": hallucinations / forbidden_total if forbidden_total else None,
        "eligible_latency_and_schema": bool(total and schema_rate >= 0.99 and p95 is not None and p95 <= 750),
        "voice_concurrency": "Run the documented live voice coexistence check while this benchmark is active.",
        "failures": failures[:20],
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the shared Gemma 4 E4B vision backend.")
    parser.add_argument("--frames", type=Path, default=Path("pc_brain/evals/vision_frames"))
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("pc_brain/data/vision_benchmark.json"))
    args = parser.parse_args()
    frames = sorted(path for path in args.frames.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not frames:
        parser.error(f"No images found in {args.frames}")
    report = await benchmark(frames, load_annotations(args.annotations))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
