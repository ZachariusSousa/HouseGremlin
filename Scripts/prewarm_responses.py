from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for and prewarm an OpenAI-compatible Responses API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--model", default="ggml-org/gemma-4-E4B-it-GGUF")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--target-seconds", type=float, default=180.0)
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/responses"
    payload = {
        "model": args.model,
        "input": [
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "You are a helpful assistant"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with the word ready."}],
            },
        ],
    }
    best_elapsed: float | None = None
    for attempt in range(1, args.attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {args.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            start = time.monotonic()
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                body = response.read()
                if response.status >= 400:
                    print(f"[prewarm][error] HTTP {response.status}: {body[:300]!r}", file=sys.stderr)
                    return 1
            elapsed = time.monotonic() - start
        except (OSError, urllib.error.URLError) as exc:
            print(f"[prewarm] attempt {attempt}/{args.attempts}: waiting for {url}: {exc}")
            if attempt < args.attempts:
                time.sleep(args.sleep)
                continue
            print(f"[prewarm][error] Responses API did not become ready at {url}", file=sys.stderr)
            return 1

        best_elapsed = elapsed if best_elapsed is None else min(best_elapsed, elapsed)
        print(f"[prewarm] attempt {attempt}/{args.attempts}: /responses took {elapsed:.1f}s")
        if elapsed <= args.target_seconds:
            print(f"[prewarm] Responses API ready for {args.model} at {args.base_url}")
            return 0
        if attempt < args.attempts:
            print("[prewarm] response was slower than target; retrying")
            time.sleep(args.sleep)

    print(
        f"[prewarm][error] /responses stayed too slow; best={best_elapsed:.1f}s target={args.target_seconds:.1f}s",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
