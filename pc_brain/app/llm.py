from dataclasses import dataclass

import httpx
from fastapi import HTTPException

from .config import Settings
from .timing import timed


SYSTEM_PROMPT = (
    "You are Robit, a small helpful home robot. "
    "Talk like rocky from the movie and book 'project hail mary"
    "Be concise and use plain spoken text only; do not use emoji, markdown, links, or stage directions."
)

ACTION_SYSTEM_PROMPT = (
    "You are Robit, a small helpful home robot with a tiny tracked body, pan/tilt head, and camera. "
    "Return only strict JSON. Do not wrap it in markdown. "
    "Use this schema: {\"response\":\"short spoken text\",\"action\":{...}}. "
    "The action object is optional. Supported action fields are "
    "movement:{direction,speed,duration_ms}, head:{pan,tilt,pan_delta,tilt_delta}, "
    "eyes:{expression,duration_ms}, emergency_stop:true. "
    "Allowed movement directions are forward, reverse, left, right, stop. "
    "Prefer short gentle movement and concise responses."
)


@dataclass
class ChatResult:
    response: str
    model: str


class OllamaChatClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _payload(self, text: str, system_prompt: str = SYSTEM_PROMPT, num_predict: int = 60) -> dict:
        return {
            "model": self.settings.llm_model,
            "think": self.settings.llm_think,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "options": {
                "num_predict": num_predict,
                "temperature": 0.4,
            },
        }

    async def _chat_with_prompt(self, text: str, system_prompt: str, num_predict: int) -> ChatResult:
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
                        json=self._payload(stripped, system_prompt, num_predict),
                    )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc

        body = response.json()
        content = body.get("message", {}).get("content", "").strip()
        if not content:
            raise HTTPException(status_code=502, detail="Ollama returned an empty response.")
        return ChatResult(response=content, model=self.settings.llm_model)

    async def chat(self, text: str) -> ChatResult:
        return await self._chat_with_prompt(text, SYSTEM_PROMPT, 60)

    async def action_chat(self, text: str) -> ChatResult:
        return await self._chat_with_prompt(text, ACTION_SYSTEM_PROMPT, 220)

    async def warmup(self) -> None:
        try:
            await self.chat("Say ready.")
        except HTTPException:
            # Warmup should not prevent robot control endpoints from starting.
            return
