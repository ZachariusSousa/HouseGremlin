from dataclasses import dataclass

import httpx
from fastapi import HTTPException

from .config import Settings
from .timing import timed


SYSTEM_PROMPT = (
    "You are Robit, a small helpful home robot. "
    "Keep spoken replies concise: one or two short sentences unless asked for detail."
)


@dataclass
class ChatResult:
    response: str
    model: str


class OllamaChatClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _payload(self, text: str) -> dict:
        return {
            "model": self.settings.llm_model,
            "think": self.settings.llm_think,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "options": {
                "num_predict": 120,
                "temperature": 0.7,
            },
        }

    async def chat(self, text: str) -> ChatResult:
        stripped = text.strip()
        if not stripped:
            raise HTTPException(status_code=400, detail="text cannot be empty")
        if self.settings.llm_provider != "ollama":
            raise HTTPException(status_code=501, detail="Only Ollama is supported for v1.")

        try:
            async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
                with timed("llm.chat.http", model=self.settings.llm_model, prompt_chars=len(stripped)):
                    response = await client.post(
                        f"{self.settings.llm_base_url}/api/chat",
                        json=self._payload(stripped),
                    )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc

        body = response.json()
        content = body.get("message", {}).get("content", "").strip()
        if not content:
            raise HTTPException(status_code=502, detail="Ollama returned an empty response.")
        return ChatResult(response=content, model=self.settings.llm_model)

    async def warmup(self) -> None:
        try:
            await self.chat("Say ready.")
        except HTTPException:
            # Warmup should not prevent robot control endpoints from starting.
            return
