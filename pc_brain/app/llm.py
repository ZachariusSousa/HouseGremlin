from dataclasses import dataclass

import httpx
from fastapi import HTTPException

from .config import Settings
from .timing import timed


SYSTEM_PROMPT = (
    "You are Robit, a small helpful home robot. "
    "Talk like Rocky from the movie and book 'Project Hail Mary'. "
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


class OpenAICompatibleChatClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _payload(self, text: str, system_prompt: str = SYSTEM_PROMPT, num_predict: int = 60) -> dict:
        payload = {
            "model": self.settings.llm_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "max_tokens": num_predict,
            "temperature": 0.4,
        }
        if self.settings.llm_think is False:
            payload["think"] = False
        return payload

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

    async def _chat_with_prompt(self, text: str, system_prompt: str, num_predict: int) -> ChatResult:
        stripped = text.strip()
        if not stripped:
            raise HTTPException(status_code=400, detail="text cannot be empty")
        if self.settings.llm_provider != "openai_compatible":
            raise HTTPException(status_code=501, detail="Only OpenAI-compatible chat is supported.")

        try:
            async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
                with timed("llm.chat.http", model=self.settings.llm_model, prompt_chars=len(stripped)):
                    response = await client.post(
                        f"{self.settings.llm_base_url}/chat/completions",
                        headers={"authorization": "Bearer local"},
                        json=self._payload(stripped, system_prompt, num_predict),
                    )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"OpenAI-compatible chat request failed: {exc}") from exc

        content = self._response_text(response.json())
        if not content:
            raise HTTPException(status_code=502, detail="OpenAI-compatible chat returned an empty response.")
        return ChatResult(response=content, model=self.settings.llm_model)

    async def chat(self, text: str) -> ChatResult:
        return await self._chat_with_prompt(text, SYSTEM_PROMPT, 60)

    async def action_chat(self, text: str) -> ChatResult:
        return await self._chat_with_prompt(text, ACTION_SYSTEM_PROMPT, 220)

    async def warmup(self) -> None:
        try:
            await self.chat("Say ready.")
        except HTTPException:
            return
