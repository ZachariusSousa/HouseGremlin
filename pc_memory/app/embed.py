"""OpenAI-compatible /v1/embeddings client with graceful unavailability.

Embeddings are optional: when the endpoint is missing or errors, `embed` returns
None and the service degrades to FTS5-only search. `/health` reports the mode
via `probe()`.
"""

from __future__ import annotations

import httpx

from pc_memory.app.config import Settings


class EmbedClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def embed(self, texts: list[str], *, timeout: float | None = None) -> list[list[float]] | None:
        """Embed a batch of texts; returns vectors or None when unavailable."""
        if not texts:
            return []
        timeout = timeout if timeout is not None else self.settings.llm_timeout
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{self.settings.llm_base_url}/embeddings",
                    headers={"authorization": "Bearer local"},
                    json={"model": self.settings.llm_model, "input": texts},
                )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            return None
        try:
            ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
            return [list(item["embedding"]) for item in ordered]
        except (KeyError, TypeError, ValueError):
            return None

    def probe(self, timeout: float = 5.0) -> bool:
        """Cheap availability check used by /health."""
        vectors = self.embed(["ping"], timeout=timeout)
        return vectors is not None and len(vectors) == 1
