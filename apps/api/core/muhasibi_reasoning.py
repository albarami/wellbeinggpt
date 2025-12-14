"""
Muḥāsibī reasoning trace (internal) + user-visible renderer.

Purpose:
- Provide a deterministic, testable representation of Al‑Muḥāsibī’s methodology
  (behavior-formation chain + staged technique) as *labels only*.
- Render the trace into a bracketed block that can be embedded into `answer_ar`
  without changing the response schema.

Safety:
- This module does NOT assert unseen causes. It only classifies using heuristics
  over the user's question and the already-available system context.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


REASONING_START = "[[MUHASIBI_REASONING_START]]"
REASONING_END = "[[MUHASIBI_REASONING_END]]"


class StimulusType(str, Enum):
    """User-described stimulus type (labels-only)."""

    INTERNAL = "internal"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class ThoughtStreamType(str, Enum):
    """Thought stream label (labels-only)."""

    INSINUATION = "insinuation"
    INSPIRATION = "inspiration"
    FLOATING_THOUGHTS = "floating_thoughts"
    SELF = "self"
    UNKNOWN = "unknown"


class ChainStage(str, Enum):
    """Behavior formation chain stage (labels-only)."""

    IDEA = "idea"
    WILL = "will"
    FIXED_INTENTION = "fixed_intention"
    ACTION = "action"
    HABIT = "habit"
    UNKNOWN = "unknown"


class TradeoffLens(str, Enum):
    """Principle-9 tradeoff lens (labels-only)."""

    SEEKING_PLEASURE = "seeking_pleasure"
    AVERSION_TO_HARDSHIP = "aversion_to_hardship"
    HOPE = "hope"
    FEAR = "fear"
    BALANCE = "balance"
    UNKNOWN = "unknown"


class MuhasibiReasoningTrace(BaseModel):
    """
    Deterministic reasoning trace for explainability.

    Notes:
    - `classification_notes_ar` is a short disclaimer about heuristic nature.
    - `evidence_used` is metadata about what the system actually retrieved.
    """

    stimulus: StimulusType = StimulusType.UNKNOWN
    thought_stream: ThoughtStreamType = ThoughtStreamType.UNKNOWN
    chain_stage: ChainStage = ChainStage.UNKNOWN
    tradeoff_lenses: list[TradeoffLens] = Field(default_factory=list)

    stage_mapping_ar: list[str] = Field(default_factory=list)
    principles_ar: list[str] = Field(default_factory=list)

    detected_entities: list[dict[str, Any]] = Field(default_factory=list)
    evidence_packets_count: int = 0
    evidence_chunk_ids: list[str] = Field(default_factory=list)

    classification_notes_ar: str = (
        "تنبيه: هذا توصيف منهجي (تصنيف تقريبي) لفهم السؤال وفق إطار المحاسبي، وليس حكمًا على حالتك."
    )


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _detect_stimulus(question_ar: str) -> StimulusType:
    q = question_ar or ""
    qn = _norm(q)
    # Heuristic: feelings/inner states -> internal, explicit events/people -> external
    if any(w in q for w in ["أشعر", "اشعر", "قلق", "خوف", "غضب", "وسواس", "خاطرة", "نية"]):
        return StimulusType.INTERNAL
    if any(w in q for w in ["مدير", "زميل", "أسرة", "عائلة", "مشكلة", "حادثة", "شائعة", "خصومة"]):
        return StimulusType.EXTERNAL
    # English keywords (some users ask bilingually)
    if any(w in qn for w in ["work", "boss", "family", "rumor", "conflict"]):
        return StimulusType.EXTERNAL
    return StimulusType.UNKNOWN


def _detect_thought_stream(question_ar: str) -> ThoughtStreamType:
    q = question_ar or ""
    if any(w in q for w in ["وسواس", "إغواء", "خاطرة سيئة", "نزغة"]):
        return ThoughtStreamType.INSINUATION
    if any(w in q for w in ["إلهام", "خاطرة طيبة", "إقبال", "اندفاع للخير"]):
        return ThoughtStreamType.INSPIRATION
    if any(w in q for w in ["خواطر", "تفكير", "أفكار", "افكار"]):
        return ThoughtStreamType.FLOATING_THOUGHTS
    if any(w in q for w in ["أنا", "نفسي", "ذاتي"]):
        return ThoughtStreamType.SELF
    return ThoughtStreamType.UNKNOWN


def _detect_chain_stage(question_ar: str) -> ChainStage:
    q = question_ar or ""
    # Habit / repetition language
    if any(w in q for w in ["عادة", "اعتدت", "متعود", "متعودة", "دائمًا", "دائما"]):
        return ChainStage.HABIT
    # Action language
    if any(w in q for w in ["كيف أطبق", "كيف اطبق", "ماذا أفعل", "ماذا افعل", "خطوات", "تطبيق"]):
        return ChainStage.ACTION
    # Fixed intention language
    if any(w in q for w in ["نية", "عزم", "قرار", "التزام"]):
        return ChainStage.FIXED_INTENTION
    # Will language
    if any(w in q for w in ["أريد", "اريد", "لا أستطيع", "لا استطيع", "أعجز", "أحاول"]):
        return ChainStage.WILL
    # Idea language
    if any(w in q for w in ["فكرة", "أفكر", "افكر", "خاطرة"]):
        return ChainStage.IDEA
    return ChainStage.UNKNOWN


def _detect_tradeoff_lenses(question_ar: str) -> list[TradeoffLens]:
    q = question_ar or ""
    lenses: list[TradeoffLens] = []
    if any(w in q for w in ["لذة", "متعة", "شهوة", "رغبة", "إغراء"]):
        lenses.append(TradeoffLens.SEEKING_PLEASURE)
    if any(w in q for w in ["مشقة", "صعب", "صعوبة", "تعب", "ألم", "الم"]):
        lenses.append(TradeoffLens.AVERSION_TO_HARDSHIP)
    if any(w in q for w in ["رجاء", "أمل", "امل", "يبشر", "أطمح"]):
        lenses.append(TradeoffLens.HOPE)
    if any(w in q for w in ["خوف", "أخاف", "اخاف", "عقوبة", "جزاء"]):
        lenses.append(TradeoffLens.FEAR)
    if not lenses:
        lenses.append(TradeoffLens.UNKNOWN)
    return lenses


def build_reasoning_trace(
    *,
    question: str,
    detected_entities: Optional[list[dict[str, Any]]] = None,
    evidence_packets: Optional[list[dict[str, Any]]] = None,
    intent: Optional[dict[str, Any]] = None,
    difficulty: Optional[str] = None,
) -> MuhasibiReasoningTrace:
    """
    Build a deterministic reasoning trace (labels-only).

    Args:
        question: Original user question (Arabic expected, but not required).
        detected_entities: Output of entity resolver (type/id/name_ar).
        evidence_packets: Retrieved evidence packets.
        intent: Optional intent classifier output.
        difficulty: Optional difficulty label.

    Returns:
        MuhasibiReasoningTrace (safe, deterministic).
    """
    q = question or ""
    ents = detected_entities or []
    packets = evidence_packets or []
    packet_ids = [str(p.get("chunk_id")) for p in packets if p.get("chunk_id")]

    # Staged technique mapping (labels)
    stage_mapping_ar = [
        "مرحلة -1: تهيئة نفسية-معرفية (إصغاء واعٍ، تحديد المقصد الأعلى، رسم طريق الهدف).",
        "مرحلة -2: استبطان/تحليل ذاتي (ملاحظة الذات، تشخيص الاتزان/الاختلال).",
        "مرحلة -3: تعديل السلوك (محاسبة/مراجعة، مراقبة الخواطر الإيجابية، تقدير الصعوبة، تدبر).",
    ]
    principles_ar = [
        "مبدأ 1: الإصغاء الواعي",
        "مبدأ 2: إدراك المقصد الأعلى للسلوك",
        "مبدأ 3: طريق بلوغ الهدف الأسمى",
        "مبدأ 4: الملاحظة الاستبطانية للذات",
        "مبدأ 6: التقويم والمراجعة (المحاسبة)",
        "مبدأ 9: التدبر/التأمل (موازنة اللذة والمشقة، والخوف والرجاء)",
    ]

    # Heuristic classification (labels only)
    trace = MuhasibiReasoningTrace(
        stimulus=_detect_stimulus(q),
        thought_stream=_detect_thought_stream(q),
        chain_stage=_detect_chain_stage(q),
        tradeoff_lenses=_detect_tradeoff_lenses(q),
        stage_mapping_ar=stage_mapping_ar,
        principles_ar=principles_ar,
        detected_entities=[dict(e) for e in ents[:10]],
        evidence_packets_count=len(packets),
        evidence_chunk_ids=packet_ids[:10],
    )

    # Add a tiny note when we have extra context, without changing meaning
    it = (intent or {}).get("intent_type") if isinstance(intent, dict) else None
    if it:
        trace.classification_notes_ar = (
            trace.classification_notes_ar
            + f" (نوع السؤال وفق مصنف النوايا: {str(it)})"
        )
    if difficulty:
        trace.classification_notes_ar = (
            trace.classification_notes_ar
            + f" (تقدير الصعوبة: {str(difficulty)})"
        )
    return trace


def render_reasoning_block(trace: MuhasibiReasoningTrace) -> str:
    """
    Render the reasoning trace into a user-visible block.

    This is designed to be embedded into `answer_ar`.
    """
    chain_ar = "مثير → فكرة → إرادة → نية ثابتة → فعل → عادة"

    def _fmt_kv(k: str, v: str) -> str:
        return f"- {k}: {v}"

    lenses = ", ".join([t.value for t in (trace.tradeoff_lenses or [])])

    lines: list[str] = [
        REASONING_START,
        "تفكير المحاسبي (عرض منهجي):",
        _fmt_kv("سلسلة تشكل السلوك", chain_ar),
        _fmt_kv("نوع المثير (تقريبًا)", trace.stimulus.value),
        _fmt_kv("سير الخواطر (تقريبًا)", trace.thought_stream.value),
        _fmt_kv("موضع السؤال في السلسلة (تقريبًا)", trace.chain_stage.value),
        _fmt_kv("عدسة التدبر (مبدأ 9)", lenses),
        _fmt_kv("عدد الأدلة المسترجعة", str(trace.evidence_packets_count)),
        _fmt_kv("معرّفات الأدلة (مختصر)", ", ".join(trace.evidence_chunk_ids[:5]) or "—"),
        "مراحل التقنية (ملخص):",
        *[f"- {s}" for s in (trace.stage_mapping_ar or [])],
        "المبادئ (ملخص):",
        *[f"- {p}" for p in (trace.principles_ar or [])],
        trace.classification_notes_ar.strip(),
        REASONING_END,
    ]
    return "\n".join([l for l in lines if l is not None]).strip() + "\n"

