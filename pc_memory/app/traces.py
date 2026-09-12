"""Trace export for the learned-router training pipeline.

`retrieval_traces` rows are the A/B training data: every retrieval stores the
question, seed ids, per-hop router decisions, visited ids and rendered context.
This module exports them as JSONL (full fidelity) or CSV (flat question ->
visited path) for offline analysis/training.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from pc_memory.app.retrieve import get_trace


def load_traces(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """All traces (id order) with JSON columns parsed back to lists."""
    rows = conn.execute("SELECT id FROM retrieval_traces ORDER BY id").fetchall()
    traces: list[dict[str, Any]] = []
    for row in rows:
        trace = get_trace(conn, row["id"])
        if trace is not None:
            traces.append(trace)
    return traces


def _flat_row(trace: dict[str, Any]) -> dict[str, Any]:
    decisions = trace.get("decisions") or []
    visited = trace.get("visited_ids") or []
    context = trace.get("context") or ""
    return {
        "trace_id": trace["id"],
        "question": trace["question"],
        "seed_ids": "|".join(str(i) for i in trace.get("seed_ids") or []),
        "visited_ids": "|".join(str(i) for i in visited),
        "hop_count": len(decisions),
        "stopped_early": bool(decisions and decisions[-1].get("stop")),
        "context_chars": len(context),
        "created_at": trace.get("created_at", ""),
    }


def export_traces(
    conn: sqlite3.Connection, path: str | Path, *, fmt: str = "jsonl"
) -> int:
    """Write all traces to `path` as JSONL or CSV; returns the row count."""
    if fmt not in ("jsonl", "csv"):
        raise ValueError(f"unsupported export format: {fmt!r} (use 'jsonl' or 'csv')")
    traces = load_traces(conn)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with out.open("w", encoding="utf-8") as fh:
            for trace in traces:
                fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
    else:
        fieldnames = [
            "trace_id",
            "question",
            "seed_ids",
            "visited_ids",
            "hop_count",
            "stopped_early",
            "context_chars",
            "created_at",
        ]
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                writer.writerow(_flat_row(trace))
    return len(traces)


__all__ = ["export_traces", "load_traces"]
