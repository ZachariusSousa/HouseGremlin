import re
from dataclasses import dataclass

import httpx
from fastapi import HTTPException
from pydantic import BaseModel

from .action_models import ActionChatOutput
from .config import Settings
from .timing import timed


LLM_PROBE_TIMEOUT_SECONDS = 5.0


SYSTEM_PROMPT = (
    "You are Robit, a small helpful home robot. "
    "Talk like Rocky from the movie and book 'Project Hail Mary'. "
    "Be concise and use plain spoken text only; do not use emoji, markdown, links, or stage directions."
)

ACTION_SYSTEM_PROMPT = (
    "You are Robit, a small helpful home robot with a tiny tracked body, pan/tilt head, and camera. "
    "Return only strict JSON. Do not wrap it in markdown. "
    "Use this schema: {\"response\":\"short spoken text\",\"action\":{...},"
    "\"vision_question\":\"question about the current camera view\"}. "
    "The action and vision_question fields are optional. When the user asks what you see, what is in front "
    "of you, or another question requiring the camera, include vision_question and do not guess the answer. "
    "Never include a movement or head action in the same response as vision_question. Supported action fields are "
    "movement:{direction,speed,duration_ms}, head:{pan,tilt,pan_delta,tilt_delta}, "
    "eyes:{expression,duration_ms}. Use movement:{direction:\"stop\"} to stop. "
    "Allowed movement directions are forward, reverse, left, right, stop. "
    "You may optionally select a temporary emotional eye expression when the message warrants it. "
    "When the user explicitly asks you to show, make, try, or change an eye expression, you must include "
    "the eyes action and must not claim it changed unless that action is present. "
    "Allowed eye expressions are neutral, angry, cute, concerned, content, happy, startled, "
    "sleepy, curious, confused, suspicious, and wink. Listening, thinking, speaking, and fault "
    "are automatic system states and must never be requested by the model. "
    "Prefer short gentle movement and concise responses."
)


@dataclass
class ChatResult:
    response: str
    model: str


class OpenAICompatibleChatClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _payload(
        self,
        text: str,
        system_prompt: str = SYSTEM_PROMPT,
        num_predict: int = 60,
        history: list[dict[str, str]] | None = None,
        response_model: type[BaseModel] | None = None,
        response_schema_name: str | None = None,
    ) -> dict:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": text})
        payload = {
            "model": self.settings.llm_model,
            "stream": False,
            "messages": messages,
            "max_tokens": num_predict,
            "temperature": 0.4,
        }
        if self.settings.llm_think is False:
            payload["think"] = False
        if response_model is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema_name or response_model.__name__,
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            }
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

    @staticmethod
    def _provider_rejected_response_format(response: httpx.Response) -> bool:
        if response.status_code not in {400, 404, 422}:
            return False
        detail = response.text.lower()
        capability_rejection_patterns = (
            r"\bresponse[_ ]format\b.{0,32}\b(?:is |are )?not supported\b",
            r"\bjson[_ ]schema\b.{0,32}\b(?:is |are )?not supported\b",
            r"\bstructured outputs?\b.{0,32}\b(?:is |are )?not supported\b",
            r"\b(?:unsupported|unknown|unrecognized)\s+(?:parameter|field)[: ]+response[_ ]format\b",
            r"\b(?:does not|doesn't) support\b.{0,32}\bresponse[_ ]format\b",
        )
        return any(re.search(pattern, detail) for pattern in capability_rejection_patterns)

    async def _chat_with_prompt(
        self,
        text: str,
        system_prompt: str,
        num_predict: int,
        history: list[dict[str, str]] | None = None,
        response_model: type[BaseModel] | None = None,
        response_schema_name: str | None = None,
    ) -> ChatResult:
        stripped = text.strip()
        if not stripped:
            raise HTTPException(status_code=400, detail="text cannot be empty")
        if self.settings.llm_provider != "openai_compatible":
            raise HTTPException(status_code=501, detail="Only OpenAI-compatible chat is supported.")

        try:
            async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
                with timed("llm.chat.http", model=self.settings.llm_model, prompt_chars=len(stripped)):
                    payload = self._payload(
                        stripped,
                        system_prompt,
                        num_predict,
                        history,
                        response_model,
                        response_schema_name,
                    )
                    response = await client.post(
                        f"{self.settings.llm_base_url}/chat/completions",
                        headers={"authorization": "Bearer local"},
                        json=payload,
                    )
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError:
                        if response_model is None or not self._provider_rejected_response_format(
                            response
                        ):
                            raise
                        response = await client.post(
                            f"{self.settings.llm_base_url}/chat/completions",
                            headers={"authorization": "Bearer local"},
                            json=self._payload(
                                stripped,
                                system_prompt,
                                num_predict,
                                history,
                            ),
                        )
                        response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"OpenAI-compatible chat request failed: {exc}") from exc

        content = self._response_text(response.json())
        if not content:
            raise HTTPException(status_code=502, detail="OpenAI-compatible chat returned an empty response.")
        return ChatResult(response=content, model=self.settings.llm_model)

    async def chat(self, text: str, history: list[dict[str, str]] | None = None) -> ChatResult:
        return await self._chat_with_prompt(text, SYSTEM_PROMPT, 60, history)

    async def action_chat(self, text: str, history: list[dict[str, str]] | None = None) -> ChatResult:
        return await self._chat_with_prompt(
            text,
            ACTION_SYSTEM_PROMPT,
            220,
            history,
            ActionChatOutput,
            "robit_action",
        )

    async def probe_models(self) -> None:
        if self.settings.llm_provider != "openai_compatible":
            raise RuntimeError("Only OpenAI-compatible chat is supported.")
        try:
            async with httpx.AsyncClient(timeout=LLM_PROBE_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    f"{self.settings.llm_base_url}/models",
                    headers={"authorization": "Bearer local"},
                )
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise RuntimeError("OpenAI-compatible model probe failed") from exc
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise RuntimeError("OpenAI-compatible model probe returned malformed data")

    async def warmup(self) -> None:
        await self.chat("Say ready.")
