"""Strict-JSON fact extraction: prompt + pydantic models, all-or-nothing.

Invalid LLM output is rejected and logged; the caller performs no store writes
when `ExtractionError` is raised (never partial writes).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from pc_memory.app.llm import LLMError

logger = logging.getLogger("pc_memory.extract")


class ExtractionError(RuntimeError):
    """Raised when the LLM output cannot be validated (whole batch rejected)."""


class Triple(BaseModel):
    """A subject–predicate–object claim; `predicate` doubles as the edge type."""

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    sentence: str | None = None  # canonical claim sentence; defaults to "subject predicate object."

    @field_validator("subject", "predicate", "object")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must be non-empty")
        return value

    @property
    def claim(self) -> str:
        return (self.sentence or f"{self.subject} {self.predicate} {self.object}.").strip()


class ExtractedFacts(BaseModel):
    triples: list[Triple] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)

    @field_validator("facts")
    @classmethod
    def _clean_facts(cls, values: list[str]) -> list[str]:
        return [v.strip() for v in values if v and v.strip()]


EXTRACTION_SYSTEM_PROMPT = (
    "You extract knowledge-graph facts from text. Return only strict JSON matching "
    '{"triples":[{"subject":"...","predicate":"snake_case_verb","object":"...",'
    '"sentence":"one declarative sentence stating the claim"}],'
    '"facts":["other important standalone sentences"]}. '
    "Use short snake_case predicates (edge types) such as lives_in, part_of, causes, "
    "located_in, example_of. Extract only what the text actually says; never invent "
    "entities or claims. If the text has no extractable facts return {\"triples\":[],\"facts\":[]}."
)


def build_extraction_prompt(text: str) -> str:
    return (
        f"Extract knowledge-graph facts from the text below.\n\n"
        f'TEXT:\n"""\n{text.strip()}\n"""\n\n'
        'Return only the strict JSON object {"triples":[...],"facts":[...]}'
    )


def validate_extraction(data: Any) -> ExtractedFacts:
    """Validate a raw dict into ExtractedFacts; raises ValidationError."""
    return ExtractedFacts.model_validate(data)


def extract_facts(client, text: str) -> ExtractedFacts:
    """Run the extraction prompt through the LLM and validate strictly.

    Raises ExtractionError (logged) when the output is invalid after the client's
    internal retry — callers must then perform no writes at all.
    """
    try:
        data = client.complete_json(
            build_extraction_prompt(text),
            system=EXTRACTION_SYSTEM_PROMPT,
            validate=validate_extraction,
        )
    except (LLMError, ValidationError) as exc:
        logger.warning("extraction rejected (%d chars of text): %s", len(text), exc)
        raise ExtractionError(f"invalid LLM extraction output: {exc}") from exc
    return validate_extraction(data)
