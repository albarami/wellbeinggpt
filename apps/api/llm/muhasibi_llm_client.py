"""
Muḥāsibī LLM Client

Wraps an LLMProvider to generate:
- PURPOSE + PATH (structured JSON)
- INTERPRET (answer from evidence packets; structured JSON)

Hard constraints:
- evidence-only for interpret
- structured JSON outputs (best-effort; validated by caller guardrails)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from apps.api.llm.gpt5_client_azure import LLMProvider, LLMRequest


_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _read_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _json_schema_for_purpose_path() -> dict[str, Any]:
    return {
        "name": "purpose_path",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "purpose": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "ultimate_goal_ar": {"type": "string"},
                        "constraints_ar": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    },
                    "required": ["ultimate_goal_ar", "constraints_ar"],
                },
                "path_plan_ar": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
            },
            "required": ["purpose", "path_plan_ar", "difficulty"],
        },
    }


def _json_schema_for_interpreter() -> dict[str, Any]:
    return {
        "name": "interpreter_output",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer_ar": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "chunk_id": {"type": "string"},
                            "source_anchor": {"type": "string"},
                            "ref": {"type": ["string", "null"]},
                        },
                        "required": ["chunk_id", "source_anchor", "ref"],
                    },
                },
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string"},
                            "id": {"type": "string"},
                            "name_ar": {"type": "string"},
                        },
                        "required": ["type", "id", "name_ar"],
                    },
                },
                "not_found": {"type": "boolean"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["answer_ar", "citations", "entities", "not_found", "confidence"],
        },
    }


@dataclass
class PurposePathResult:
    ultimate_goal_ar: str
    constraints_ar: list[str]
    path_plan_ar: list[str]
    difficulty: str


@dataclass
class InterpretResult:
    answer_ar: str
    citations: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    not_found: bool
    confidence: str


class MuhasibiLLMClient:
    """High-level Muḥāsibī LLM client."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def purpose_path(self, question: str) -> Optional[PurposePathResult]:
        system_prompt = _read_prompt("purpose_path.md")
        req = LLMRequest(
            system_prompt=system_prompt,
            user_message=question,
            response_format=_json_schema_for_purpose_path(),
            temperature=0.2,
            max_tokens=800,
        )
        resp = await self.provider.complete(req)
        if resp.error:
            return None
        data = resp.parsed_json
        if not isinstance(data, dict):
            # Best-effort parse if structured output wasn't respected
            try:
                data = json.loads(resp.content)
            except Exception:
                return None
        try:
            purpose = data["purpose"]
            return PurposePathResult(
                ultimate_goal_ar=purpose["ultimate_goal_ar"],
                constraints_ar=list(purpose["constraints_ar"]),
                path_plan_ar=list(data["path_plan_ar"]),
                difficulty=str(data["difficulty"]),
            )
        except Exception:
            return None

    async def interpret(
        self,
        question: str,
        evidence_packets: list[dict[str, Any]],
        detected_entities: list[dict[str, Any]],
    ) -> Optional[InterpretResult]:
        system_prompt = _read_prompt("interpreter.md")
        user_payload = {
            "question": question,
            "evidence_packets": evidence_packets,
            "detected_entities": detected_entities,
        }
        req = LLMRequest(
            system_prompt=system_prompt,
            user_message=json.dumps(user_payload, ensure_ascii=False),
            response_format=_json_schema_for_interpreter(),
            temperature=0.2,
            max_tokens=1200,
        )
        resp = await self.provider.complete(req)
        if resp.error:
            return None
        data = resp.parsed_json
        if not isinstance(data, dict):
            try:
                data = json.loads(resp.content)
            except Exception:
                return None
        try:
            return InterpretResult(
                answer_ar=str(data["answer_ar"]),
                citations=list(data["citations"]),
                entities=list(data["entities"]),
                not_found=bool(data["not_found"]),
                confidence=str(data["confidence"]),
            )
        except Exception:
            return None


