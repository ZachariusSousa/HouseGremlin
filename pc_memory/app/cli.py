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
    python -m pc_memory.app.cli export-traces --out traces.jsonl [--format csv]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from pc_memory.app.brain_ingest import (
    DEFAULT_BRAIN_DB,
    DEFAULT_STATE_FILE,
    ingest_brain,
    watch_brain,
)
from pc_memory.app.config import load_settings
from pc_memory.app.db import connect, init_schema, rebuild_fts
from pc_memory.app.embed import EmbedClient
from pc_memory.app.ingest import IngestError, ingest_text, ingest_url
from pc_memory.app.llm import ChatClient, LLMError
from pc_memory.app.retrieve import retrieve
from pc_memory.app.research import (
    load_research_state,
    research_topic,
)
from pc_memory.app.seed import seed_sky_chain
from pc_memory.app.store import add_fact, forget_node, inspect_node, re_embed, stats
from pc_memory.app.traces import export_traces


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
            print(
                f"warning: LLM unavailable ({exc}); using exact-match dedup only",
                file=sys.stderr,
            )
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
            conn,
            args.url,
            llm=ChatClient(settings),
            embed=EmbedClient(settings),
            confidence=args.confidence,
        )
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(json.dumps(result))
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    settings, conn = _open(args)
    try:
        result = seed_sky_chain(conn, embed=EmbedClient(settings))
    finally:
        conn.close()
    print(json.dumps(result, indent=2))
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    settings, conn = _open(args)
    try:
        embed = EmbedClient(settings)
        try:
            result = retrieve(
                conn,
                args.question,
                llm=ChatClient(settings),
                embed=embed,
                max_hops=args.max_hops,
                beam=args.beam,
                budget_chars=settings.context_budget_chars,
            )
        except LLMError as exc:
            print(
                f"warning: LLM unavailable ({exc}); seed-only retrieval",
                file=sys.stderr,
            )
            result = retrieve(
                conn,
                args.question,
                llm=None,
                embed=embed,
                max_hops=args.max_hops,
                beam=args.beam,
                budget_chars=settings.context_budget_chars,
            )
    finally:
        conn.close()
    print(json.dumps(result, indent=2))
    return 0


def cmd_export_traces(args: argparse.Namespace) -> int:
    _settings, conn = _open(args)
    try:
        count = export_traces(conn, args.out, fmt=args.format)
    finally:
        conn.close()
    print(json.dumps({"exported": count, "path": str(args.out), "format": args.format}))
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


def cmd_re_embed(args: argparse.Namespace) -> int:
    settings, conn = _open(args)
    try:
        result = re_embed(conn, EmbedClient(settings))
    finally:
        conn.close()
    print(json.dumps(result))
    return 0 if result.get("failed_at_batch") is None else 1


def cmd_forget(args: argparse.Namespace) -> int:
    _settings, conn = _open(args)
    try:
        result = forget_node(conn, args.node_id)
    finally:
        conn.close()
    print(json.dumps(result))
    return 0


def _resolve_brain_db(settings, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    # Default: <repo root>/pc_brain/data/brain.db (package root is pc_memory/).
    return settings.db_path.parent.parent.parent / "pc_brain" / "data" / "brain.db"


def _resolve_state_file(settings, explicit: str | None) -> Path:
    """Default state file sits next to the memory DB (not CWD-relative)."""
    if explicit:
        return Path(explicit).expanduser()
    return settings.db_path.parent / DEFAULT_STATE_FILE


def _resolve_research_state(settings, explicit: str | None) -> Path:
    """Research BFS state also lives next to the memory DB."""
    if explicit:
        return Path(explicit).expanduser()
    return settings.db_path.parent / "research_state.json"


def cmd_ingest_brain(args: argparse.Namespace) -> int:
    settings, conn = _open(args)
    try:
        embed = None if args.no_embed else EmbedClient(settings)
        summary = ingest_brain(
            _resolve_brain_db(settings, args.brain_db),
            _resolve_state_file(settings, args.state_file),
            conn,
            embed=embed,
            confidence=args.confidence,
            dry_run=args.dry_run,
        )
    finally:
        conn.close()
    print(json.dumps(summary, indent=2))
    return 0


def cmd_watch_brain(args: argparse.Namespace) -> int:
    settings, conn = _open(args)
    try:
        embed = None if args.no_embed else EmbedClient(settings)
        watch_brain(
            _resolve_brain_db(settings, args.brain_db),
            _resolve_state_file(settings, args.state_file),
            conn,
            embed=embed,
            confidence=args.confidence,
            poll_interval=args.poll_seconds,
        )
    finally:
        conn.close()
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    settings, conn = _open(args)
    try:
        state_path = _resolve_research_state(settings, args.state_file)
        if args.reset and state_path.exists():
            state_path.unlink()
        embed = None if args.no_embed else EmbedClient(settings)
        summary = research_topic(
            args.subject,
            conn,
            llm=ChatClient(settings),
            state_path=_resolve_research_state(settings, args.state_file),
            embed=embed,
            confidence=args.confidence,
            max_depth=args.max_depth,
            breadth_per_level=args.breadth,
            max_sources=args.max_sources,
        )
    finally:
        conn.close()
    print(json.dumps(summary, indent=2))
    return 0


def cmd_watch_research(args: argparse.Namespace) -> int:
    """Keep expanding the frontier until it's empty (the 'while I'm away' mode)."""
    settings, conn = _open(args)
    try:
        embed = None if args.no_embed else EmbedClient(settings)
        state_path = _resolve_research_state(settings, args.state_file)
        while True:
            state = load_research_state(state_path)
            if state is None or not state.frontier:
                print("frontier exhausted (or no research started); nothing to do.")
                break
            summary = research_topic(
                state.seed or args.subject,
                conn,
                llm=ChatClient(settings),
                state_path=state_path,
                embed=embed,
                confidence=args.confidence,
                max_depth=1,
                breadth_per_level=args.breadth,
                max_sources=args.max_sources,
            )
            print(json.dumps({"round": summary["totals"], "frontier_remaining": summary["frontier_remaining"]}))
            if args.poll_seconds > 0:
                time.sleep(args.poll_seconds)
    finally:
        conn.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pc_memory", description="SQLite knowledge-graph memory service CLI"
    )
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

    p = sub.add_parser(
        "ingest-url", help="Fetch a URL (cached) and LLM-extract triples/facts"
    )
    p.add_argument("url")
    p.add_argument("--confidence", type=float, default=0.7)
    p.set_defaults(func=cmd_ingest_url)

    p = sub.add_parser(
        "seed", help="Seed the sky-is-blue causal chain + distractors (idempotent)"
    )
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser(
        "retrieve", help="Routed retrieval for a question (writes a trace row)"
    )
    p.add_argument("question")
    p.add_argument("--max-hops", type=int, default=None)
    p.add_argument("--beam", type=int, default=None)
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("inspect", help="Show a node with its edges and provenance")
    p.add_argument("node_id", type=int)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("stats", help="Node/edge/provenance counts")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("health", help="Health + embedding mode")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("rebuild", help="Re-sync the FTS5 index from nodes")
    p.set_defaults(func=cmd_rebuild)

    p = sub.add_parser(
        "re-embed",
        help="Regenerate all node embeddings with the current model (run after changing embedding backend)",
    )
    p.set_defaults(func=cmd_re_embed)

    p = sub.add_parser(
        "export-traces",
        help="Export retrieval traces (JSONL or CSV) for router training",
    )
    p.add_argument("--out", required=True, help="Output file path")
    p.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    p.set_defaults(func=cmd_export_traces)

    p = sub.add_parser("forget", help="Delete a node and cascade edges/provenance")
    p.add_argument("node_id", type=int)
    p.set_defaults(func=cmd_forget)

    common_brain = [
        ("--brain-db", dict(default=None, help=f"Path to brain.db (default: {DEFAULT_BRAIN_DB} under repo root)")),
        # None (not a string) so the resolver picks the DB-adjacent default path.
        ("--state-file", dict(default=None, help="Ingestion state file (default: next to the memory DB)")),
        ("--confidence", dict(type=float, default=0.7)),
        ("--no-embed", dict(action="store_true", help="Skip embeddings (FTS5-only mode)")),
    ]

    p = sub.add_parser(
        "ingest-brain",
        help="One-shot: ingest new robot events from brain.db into the memory graph (no LLM)",
    )
    for name, opts in common_brain:
        p.add_argument(name, **opts)
    p.add_argument("--dry-run", action="store_true", help="Show claims without writing to the store")
    p.set_defaults(func=cmd_ingest_brain)

    p = sub.add_parser(
        "watch-brain",
        help="Background: poll brain.db for new events and ingest them continuously (no LLM)",
    )
    for name, opts in common_brain:
        p.add_argument(name, **opts)
    p.add_argument("--poll-seconds", type=float, default=15.0, help="Poll interval (default 15s)")
    p.set_defaults(func=cmd_watch_brain)

    common_research = [
        ("--state-file", dict(default=None, help="Research BFS state file (default: next to the memory DB)")),
        ("--confidence", dict(type=float, default=0.6)),
        ("--no-embed", dict(action="store_true", help="Skip embeddings (FTS5-only mode)")),
        ("--max-sources", dict(type=int, default=3, help="Web sources fetched per subject (default 3)")),
        ("--breadth", dict(type=int, default=5, help="Subjects researched per BFS level (default 5)")),
        ("--reset", dict(action="store_true", help="Delete the research state file before starting")),
    ]

    p = sub.add_parser(
        "research",
        help="Grow the knowledge graph from a subject: web search + LLM extraction + related-subject frontier",
    )
    p.add_argument("subject")
    for name, opts in common_research:
        p.add_argument(name, **opts)
    p.add_argument("--max-depth", type=int, default=1, help="BFS levels beyond the seed (default 1)")
    p.set_defaults(func=cmd_research)

    p = sub.add_parser(
        "watch-research",
        help="Keep expanding the research frontier until it's empty (runs while you're away)",
    )
    p.add_argument("subject", nargs="?", default=None, help="Seed subject (only used if no state file exists)")
    for name, opts in common_research:
        p.add_argument(name, **opts)
    p.add_argument("--poll-seconds", type=float, default=5.0, help="Pause between rounds (default 5s; 0 = none)")
    p.set_defaults(func=cmd_watch_research)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func  # type: ignore[assignment]
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
