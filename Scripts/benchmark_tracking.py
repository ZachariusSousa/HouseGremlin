from __future__ import annotations

import argparse
import json
import statistics
import time
from io import BytesIO

import httpx
from PIL import Image


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return ordered[index]


def rotate_jpeg(content: bytes) -> bytes:
    with Image.open(BytesIO(content)) as source:
        output = BytesIO()
        source.convert("RGB").rotate(180, expand=True).save(output, format="JPEG", quality=85)
        return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Robit's local RF-DETR Nano sidecar.")
    parser.add_argument("--brain-url", default="http://127.0.0.1:8080")
    parser.add_argument("--tracking-url", default="http://127.0.0.1:8091")
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()

    request_latencies: list[float] = []
    detector_latencies: list[float] = []
    errors: list[str] = []
    frames_with_people = 0

    with httpx.Client(timeout=10.0) as client:
        health = client.get(f"{args.tracking_url.rstrip('/')}/health")
        health.raise_for_status()
        health_payload = health.json()
        if not health_payload.get("available"):
            print(json.dumps({"available": False, "reason": health_payload.get("reason")}, indent=2))
            return 1
        for _ in range(max(1, args.count)):
            try:
                frame = client.get(f"{args.brain_url.rstrip('/')}/robot/camera/capture?fresh=true")
                frame.raise_for_status()
                frame_id = frame.headers["x-robit-frame-id"]
                captured_at = frame.headers["x-robit-captured-at"]
                content = rotate_jpeg(frame.content)
                started = time.perf_counter()
                response = client.post(
                    f"{args.tracking_url.rstrip('/')}/detect",
                    content=content,
                    headers={
                        "Content-Type": "image/jpeg",
                        "X-Robit-Frame-Id": frame_id,
                        "X-Robit-Captured-At": captured_at,
                        "X-Robit-Threshold": "0.40",
                    },
                )
                request_latencies.append((time.perf_counter() - started) * 1000.0)
                response.raise_for_status()
                payload = response.json()
                detector_latencies.append(float(payload["latency_ms"]))
                frames_with_people += bool(payload.get("people"))
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                errors.append(str(exc))

    report = {
        "backend": health_payload.get("backend"),
        "model": health_payload.get("model"),
        "frames_requested": max(1, args.count),
        "frames_completed": len(detector_latencies),
        "frames_with_people": frames_with_people,
        "request_latency_ms": {
            "mean": statistics.fmean(request_latencies) if request_latencies else None,
            "p50": percentile(request_latencies, 0.50),
            "p95": percentile(request_latencies, 0.95),
        },
        "detector_latency_ms": {
            "mean": statistics.fmean(detector_latencies) if detector_latencies else None,
            "p50": percentile(detector_latencies, 0.50),
            "p95": percentile(detector_latencies, 0.95),
        },
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
