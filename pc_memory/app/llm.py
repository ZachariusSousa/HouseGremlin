"""OpenAI-compatible chat client for pc_memory (pattern from pc_brain/app/llm.py).

MVP note: this client is synchronous (httpx.Client) so the same code path serves
the FastAPI endpoints (sync endpoint functions), the CLI and the tests. The
payload/response shape mirrors pc_brain's async client; `think: false` is always
sent (llama-server kwarg, ignored by strict OpenAI servers that accept extras).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import httpx

from pc_memory.app.config import Settings

JSON_SYSTEM_PROMPT = (
    "You are a precise knowledge-graph assistant. "
    "Return only strict JSON. Do not wrap it in markdown or code fences. "
    "No comments, no trailing commas, no text before or after the JSON object."
)


class LLMError(RuntimeError):
    """Raised when the LLM is unreachable or keeps returning invalid output."""


def extract_json_object(raw: str) -> dict | None:
    """Parse a strict-JSON object out of raw LLM text.

    Tolerates code fences and surrounding prose by locating the outermost
    braces; returns None when no JSON object can be parsed.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class ChatClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _payload(self, prompt: str, system: str, max_tokens: int, temperature: float) -> dict:
        return {
            "model": self.settings.llm_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "think": False,
        }

    @staticmethod
    def _response_text(body: dict) -> str:
        choices = body.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
                ).strip()
        output = body.get("output") or []
        parts = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if isinstance(content, dict) and content.get("type") in {"text", "output_text"}:
                    parts.append(content.get("text", ""))
        return " ".join(parts).strip()

    def chat(
        self,
        prompt: str,
        *,
        system: str = JSON_SYSTEM_PROMPT,
        max_tokens: int = 1200,
        temperature: float = 0.2,
    ) -> str:
        """Single chat completion; returns the raw response text."""
        stripped = prompt.strip()
        if not stripped:
            raise LLMError("prompt is empty")
        try:
            with httpx.Client(timeout=self.settings.llm_timeout) as client:
                response = client.post(
                    f"{self.settings.llm_base_url}/chat/completions",
                    headers={"authorization": "Bearer local"},
                    json=self._payload(stripped, system, max_tokens, temperature),
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"chat request failed: {exc}") from exc
        content = self._response_text(response.json())
        if not content:
            raise LLMError("chat returned an empty response")
        return content

    def complete_json(
        self,
        prompt: str,
        *,
        system: str = JSON_SYSTEM_PROMPT,
        max_tokens: int = 1200,
        validate: Callable[[dict], Any] | None = None,
    ) -> dict:
        """Chat until a strict-JSON object comes back; one retry on parse failure.

        `validate` (optional) is called with the parsed dict; if it raises, the
        attempt counts as a parse failure and triggers the single retry.
        Raises LLMError when both attempts fail.
        """
        last_error = "no response"
        for attempt in range(2):
            attempt_prompt = prompt
            if attempt > 0:
                attempt_prompt = (
                    f"{prompt}\n\nYour previous response was not valid JSON. "
                    "Respond again with only the strict JSON object."
                )
            try:
                raw = self.chat(attempt_prompt, system=system, max_tokens=max_tokens)
                data = extract_json_object(raw)
                if data is None:
                    raise ValueError(f"no JSON object in response: {raw[:200]!r}")
                if validate is not None:
                    validate(data)
                return data
            except (ValueError, LLMError) as exc:
                last_error = str(exc)
        raise LLMError(f"LLM returned invalid JSON after retry: {last_error}")
