"""Web search (ddgs/DuckDuckGo) + page fetch (httpx + BeautifulSoup).

Failures are skip-and-continue: individual URLs that fail are logged and
skipped, never fatal. Fetched pages are cached in the `url_cache` table so a
URL is downloaded at most once per database. A simple module-level rate limit
spaces out downloads.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

logger = logging.getLogger("pc_memory.search")

MIN_FETCH_INTERVAL_SECONDS = 1.0
_rate_lock = threading.Lock()
_last_fetch_at = 0.0


def _rate_limit() -> None:
    global _last_fetch_at
    with _rate_lock:
        wait = MIN_FETCH_INTERVAL_SECONDS - (time.monotonic() - _last_fetch_at)
        if wait > 0:
            time.sleep(wait)
        _last_fetch_at = time.monotonic()


def web_search(query: str, max_results: int = 5) -> list[str]:
    """DuckDuckGo text search via ddgs; returns deduped URLs (order preserved)."""
    try:
        rows = DDGS().text(query, max_results=max_results) or []
    except Exception as exc:  # ddgs raises assorted network/parse errors
        logger.warning("web_search(%r) failed: %s", query, exc)
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = row.get("url") or row.get("href") or ""
        url = str(url).strip()
        if url.startswith("http") and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls[:max_results]


def html_to_text(html: str) -> str:
    """Render HTML to plain text (scripts/styles dropped, whitespace collapsed)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def _download(url: str, timeout: float) -> str:
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url, headers={"user-agent": "pc_memory/0.1"})
    response.raise_for_status()
    return response.text


def fetch_url(conn, url: str, *, timeout: float = 15.0) -> str | None:
    """Fetch one URL to plain text, using the per-URL cache table.

    Returns cached/fetched text, or None when the fetch fails (skip-and-continue).
    """
    row = conn.execute("SELECT content FROM url_cache WHERE url = ?", (url,)).fetchone()
    if row is not None:
        return row["content"]
    _rate_limit()
    try:
        html = _download(url, timeout)
    except Exception as exc:
        logger.warning("fetch_url(%s) failed, skipping: %s", url, exc)
        return None
    text = html_to_text(html)
    if not text:
        return None
    conn.execute(
        "INSERT OR REPLACE INTO url_cache (url, content, fetched_at) VALUES (?, ?, datetime('now'))",
        (url, text),
    )
    conn.commit()
    return text


def fetch_urls(conn, urls: list[str], *, timeout: float = 15.0) -> dict[str, str]:
    """Fetch several URLs; returns {url: text} for the ones that succeeded."""
    out: dict[str, str] = {}
    for url in urls:
        text = fetch_url(conn, url, timeout=timeout)
        if text:
            out[url] = text
    return out
