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
from uuid import UUID

from apps.api.llm.gpt5_client_azure import LLMProvider, LLMRequest


def _sanitize_for_json(obj: Any) -> Any:
    """
    Convert non-JSON-serializable objects to strings.
    
    This handles UUIDs and other types that can't be directly serialized.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, UUID):
        return str(obj)
    elif hasattr(obj, "__dict__"):
        return _sanitize_for_json(vars(obj))
    return obj


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


def _json_schema_for_query_rewrite_ar() -> dict[str, Any]:
    return {
        "name": "query_rewrite_ar",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rewrites_ar": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                },
                "focus_terms_ar": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 8,
                },
                "disambiguation_question_ar": {"type": ["string", "null"]},
            },
            "required": ["rewrites_ar", "focus_terms_ar", "disambiguation_question_ar"],
        },
    }


def _json_schema_for_intent_classifier_ar() -> dict[str, Any]:
    return {
        "name": "intent_classifier_ar",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "intent_type": {
                    "type": "string",
                    "enum": [
                        "list_pillars",
                        "list_core_values_in_pillar",
                        "list_sub_values_in_core_value",
                        "definition",
                        "definition_with_evidence",
                        "compare",
                        "connect_across_pillars",
                        "practical_guidance",
                        "out_of_scope_fiqh_ruling",
                        "out_of_scope_biography",
                        "out_of_scope_general_knowledge",
                        "ambiguous",
                    ],
                },
                "is_in_scope": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "target_entity_type": {"type": ["string", "null"]},
                "target_entity_name_ar": {"type": ["string", "null"]},
                "notes_ar": {"type": "string"},
                "suggested_queries_ar": {"type": "array", "items": {"type": "string"}},
                "required_clarification_question_ar": {"type": ["string", "null"]},
            },
            "required": [
                "intent_type",
                "is_in_scope",
                "confidence",
                "target_entity_type",
                "target_entity_name_ar",
                "notes_ar",
                "suggested_queries_ar",
                "required_clarification_question_ar",
            ],
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
        mode: str = "answer",
        used_edges: Optional[list[dict[str, Any]]] = None,
        argument_chains: Optional[list[dict[str, Any]]] = None,
        fallback_context: Optional[dict[str, Any]] = None,
    ) -> Optional[InterpretResult]:
        prompt_name = "interpreter.md"
        if mode == "debate":
            prompt_name = "interpreter_debate_ar.md"
        elif mode == "socratic":
            prompt_name = "interpreter_socratic_ar.md"
        elif mode == "judge":
            prompt_name = "interpreter_judge_ar.md"
        elif mode == "natural_chat":
            prompt_name = "interpreter_natural_chat_ar.md"

        system_prompt = _read_prompt(prompt_name)
        user_payload = {
            "question": question,
            "evidence_packets": _sanitize_for_json(evidence_packets),
            "detected_entities": _sanitize_for_json(detected_entities),
            "mode": mode,
            "used_edges": _sanitize_for_json(used_edges or []),
            "argument_chains": _sanitize_for_json(argument_chains or []),
            "fallback_context": _sanitize_for_json(fallback_context or {}),
        }
        # Natural chat needs more tokens for flowing scholarly prose
        tokens = 2000 if mode == "natural_chat" else 1200
        req = LLMRequest(
            system_prompt=system_prompt,
            user_message=json.dumps(user_payload, ensure_ascii=False),
            response_format=_json_schema_for_interpreter(),
            temperature=0.3 if mode == "natural_chat" else 0.2,
            max_tokens=tokens,
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

    async def query_rewrite_ar(
        self,
        question: str,
        detected_entities: list[dict[str, Any]],
        keywords: list[str],
    ) -> Optional[dict[str, Any]]:
        """
        Generate Arabic query rewrites to improve retrieval.

        Safety:
        - This method must NOT answer the question.
        - Output is used only to run additional retrieval queries.
        """
        system_prompt = _read_prompt("query_rewrite_ar.md")
        user_payload = {
            "question": question,
            "detected_entities": [
                {"type": e.get("type"), "name_ar": e.get("name_ar"), "confidence": e.get("confidence", 0.0)}
                for e in (detected_entities or [])[:8]
            ],
            "keywords": (keywords or [])[:12],
        }
        req = LLMRequest(
            system_prompt=system_prompt,
            user_message=json.dumps(user_payload, ensure_ascii=False),
            response_format=_json_schema_for_query_rewrite_ar(),
            temperature=0.2,
            max_tokens=500,
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

        if not isinstance(data, dict):
            return None

        # Basic structural validation
        if not isinstance(data.get("rewrites_ar"), list) or not data["rewrites_ar"]:
            return None
        if not isinstance(data.get("focus_terms_ar"), list):
            data["focus_terms_ar"] = []
        if "disambiguation_question_ar" not in data:
            data["disambiguation_question_ar"] = None

        return data

    async def classify_intent_ar(
        self,
        question: str,
        detected_entities: list[dict[str, Any]],
        keywords: list[str],
    ) -> Optional[dict[str, Any]]:
        """
        Classify user intent in Arabic (structured) without answering.
        """
        system_prompt = _read_prompt("intent_classifier_ar.md")
        user_payload = {
            "question": question,
            "detected_entities": _sanitize_for_json(detected_entities),
            "keywords": _sanitize_for_json(keywords),
        }
        req = LLMRequest(
            system_prompt=system_prompt,
            user_message=json.dumps(user_payload, ensure_ascii=False),
            response_format=_json_schema_for_intent_classifier_ar(),
            temperature=0.0,
            max_tokens=400,
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
        # Basic shape guard
        if "intent_type" not in data or "is_in_scope" not in data:
            return None
        return data


