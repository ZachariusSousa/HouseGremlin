"""Command-line interface for pc_memory.

Examples (from repo root, using the package venv):

    python -m pc_memory.app.cli add-fact "The sky is blue." --confidence 0.9
    python -m pc_memory.app.cli ingest-text "Rayleigh scattering ..." --source-kind manual
    python -m pc_memory.app.cli ingest-url https://example.com/rayleigh
    python -m pc_memory.app.cli inspect 42
    python -m pc_memory.app.cli stats
    python -m pc_memory.app.cli health
    python -m pc_memory.app.cli rebuild
    python -m pc_memory.app.cli forget 42
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

from pc_memory.app.config import load_settings
from pc_memory.app.db import connect, init_schema, rebuild_fts
from pc_memory.app.embed import EmbedClient
from pc_memory.app.ingest import IngestError, ingest_text, ingest_url
from pc_memory.app.llm import ChatClient, LLMError
from pc_memory.app.store import add_fact, forget_node, inspect_node, stats


def _open(args: argparse.Namespace):
    settings = load_settings()
    conn = connect(settings.db_path)
    init_schema(conn)
    return settings, conn


def _judge_or_warn(llm: ChatClient) -> Callable[[str], str]:
    """LLM verdict client for the CLI; degrades to exact-match dedup when the LLM is down."""

    def judge(prompt: str) -> str:
        try:
            return llm.chat(prompt)
        except LLMError as exc:
            print(f"warning: LLM unavailable ({exc}); using exact-match dedup only", file=sys.stderr)
            return ""

    return judge


def cmd_add_fact(args: argparse.Namespace) -> int:
    settings, conn = _open(args)
    try:
        result = add_fact(
            conn,
            args.text,
            confidence=args.confidence,
            provenance={
                "source_kind": args.source_kind,
                "source_ref": args.source_ref,
                "snippet": args.text[:500],
            },
            judge=_judge_or_warn(ChatClient(settings)),
        )
    finally:
        conn.close()
    print(json.dumps({"node_id": result.node_id, "verdict": result.verdict}))
    return 0


def cmd_ingest_text(args: argparse.Namespace) -> int:
    settings, conn = _open(args)
    try:
        result = ingest_text(
            conn,
            args.text,
            llm=ChatClient(settings),
            embed=EmbedClient(settings),
            confidence=args.confidence,
            source_kind=args.source_kind,
            source_ref=args.source_ref,
        )
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(json.dumps(result))
    return 0


def cmd_ingest_url(args: argparse.Namespace) -> int:
    settings, conn = _open(args)
    try:
        result = ingest_url(
            conn, args.url, llm=ChatClient(settings), embed=EmbedClient(settings), confidence=args.confidence
        )
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(json.dumps(result))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    _settings, conn = _open(args)
    try:
        info = inspect_node(conn, args.node_id)
    finally:
        conn.close()
    if info is None:
        print(f"error: node {args.node_id} not found", file=sys.stderr)
        return 1
    print(json.dumps(info, indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    _settings, conn = _open(args)
    try:
        data = stats(conn)
    finally:
        conn.close()
    print(json.dumps(data, indent=2))
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    settings, conn = _open(args)
    try:
        mode = "hybrid" if EmbedClient(settings).probe() else "fts5_only"
    finally:
        conn.close()
    print(json.dumps({"status": "ok", "embedding_mode": mode}))
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    _settings, conn = _open(args)
    try:
        rebuild_fts(conn)
    finally:
        conn.close()
    print(json.dumps({"fts_rebuilt": True}))
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    _settings, conn = _open(args)
    try:
        result = forget_node(conn, args.node_id)
    finally:
        conn.close()
    print(json.dumps(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pc_memory", description="SQLite knowledge-graph memory service CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add-fact", help="Add one canonical fact node with provenance")
    p.add_argument("text")
    p.add_argument("--confidence", type=float, default=0.7)
    p.add_argument("--source-kind", default="manual")
    p.add_argument("--source-ref", default=None)
    p.set_defaults(func=cmd_add_fact)

    p = sub.add_parser("ingest-text", help="LLM-extract triples/facts from raw text")
    p.add_argument("text")
    p.add_argument("--confidence", type=float, default=0.7)
    p.add_argument("--source-kind", default="manual")
    p.add_argument("--source-ref", default=None)
    p.set_defaults(func=cmd_ingest_text)

    p = sub.add_parser("ingest-url", help="Fetch a URL (cached) and LLM-extract triples/facts")
    p.add_argument("url")
    p.add_argument("--confidence", type=float, default=0.7)
    p.set_defaults(func=cmd_ingest_url)

    p = sub.add_parser("inspect", help="Show a node with its edges and provenance")
    p.add_argument("node_id", type=int)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("stats", help="Node/edge/provenance counts")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("health", help="Health + embedding mode")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("rebuild", help="Re-sync the FTS5 index from nodes")
    p.set_defaults(func=cmd_rebuild)

    p = sub.add_parser("forget", help="Delete a node and cascade edges/provenance")
    p.add_argument("node_id", type=int)
    p.set_defaults(func=cmd_forget)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func  # type: ignore[assignment]
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
