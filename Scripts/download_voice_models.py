from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Hugging Face voice sidecar models with visible progress.")
    parser.add_argument("repo_ids", nargs="+", help="Hugging Face repo ids to download.")
    args = parser.parse_args()

    for repo_id in args.repo_ids:
        print(f"[download-models] downloading {repo_id}")
        path = Path(snapshot_download(repo_id=repo_id))
        print(f"[download-models] ready {repo_id}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
