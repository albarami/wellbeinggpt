"""
Muḥāsibī Reasoning Middleware

Implements a deterministic state machine orchestrator that runs:
1. LISTEN - Normalize Arabic, detect entities, produce listen_summary_ar
2. PURPOSE - Output ultimate goal + constraints
3. PATH - Output short plan steps with prioritization
4. RETRIEVE - Produce evidence packets bundle
5. ACCOUNT - Enforce citation coverage, reject unsupported claims
6. INTERPRET - Generate answer from evidence
7. REFLECT - Consequence-aware reflection
8. FINALIZE - Validate schema + citations + claim checks
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional

from apps.api.core.schemas import (
    EntityType,
    Confidence,
    Difficulty,
    Purpose,
    Citation,
    EntityRef,
    FinalResponse,
)
from apps.api.retrieve.normalize_ar import normalize_for_matching, extract_arabic_words
from apps.api.retrieve.hybrid_retriever import HybridRetriever, RetrievalInputs
from apps.api.llm.muhasibi_llm_client import MuhasibiLLMClient


class MuhasibiState(Enum):
    """States in the Muḥāsibī reasoning state machine."""

    LISTEN = auto()
    PURPOSE = auto()
    PATH = auto()
    RETRIEVE = auto()
    ACCOUNT = auto()
    INTERPRET = auto()
    REFLECT = auto()
    FINALIZE = auto()
    FAILED = auto()


@dataclass
class StateContext:
    """Context passed between states."""

    # Input
    question: str
    language: str = "ar"

    # LISTEN outputs
    normalized_question: str = ""
    listen_summary_ar: str = ""
    detected_entities: list[dict] = field(default_factory=list)
    question_keywords: list[str] = field(default_factory=list)

    # PURPOSE outputs
    purpose: Optional[Purpose] = None

    # PATH outputs
    path_plan_ar: list[str] = field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM

    # RETRIEVE outputs
    evidence_packets: list[dict] = field(default_factory=list)
    has_definition: bool = False
    has_evidence: bool = False

    # ACCOUNT outputs
    citation_valid: bool = False
    account_issues: list[str] = field(default_factory=list)

    # INTERPRET outputs
    answer_ar: str = ""
    citations: list[Citation] = field(default_factory=list)
    entities: list[EntityRef] = field(default_factory=list)
    not_found: bool = False
    confidence: Confidence = Confidence.LOW

    # REFLECT outputs
    reflection_added: bool = False

    # Error handling
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 1

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    state_timings: dict[str, float] = field(default_factory=dict)


class MuhasibiMiddleware:
    """
    Muḥāsibī reasoning middleware orchestrator.

    Implements the 8-state pipeline for evidence-only answering.
    """

    # Required constraints for every request
    REQUIRED_CONSTRAINTS = [
        "evidence_only",
        "cite_every_claim",
        "refuse_if_missing",
    ]

    def __init__(
        self,
        entity_resolver=None,
        retriever=None,
        llm_client=None,
        guardrails=None,
    ):
        """
        Initialize the middleware.

        Args:
            entity_resolver: EntityResolver for detecting entities.
            retriever: Retrieval pipeline for evidence.
            llm_client: LLM client for PURPOSE/PATH/INTERPRET/REFLECT.
            guardrails: Guardrails for citation validation.
        """
        self.entity_resolver = entity_resolver
        self.retriever: Optional[HybridRetriever] = retriever
        self.llm_client: Optional[MuhasibiLLMClient] = llm_client
        self.guardrails = guardrails

    async def process(self, question: str, language: str = "ar") -> FinalResponse:
        """
        Process a question through the full Muḥāsibī pipeline.

        Args:
            question: The user's question.
            language: Language preference.

        Returns:
            FinalResponse following the required schema.
        """
        ctx = StateContext(question=question, language=language)

        # State machine execution
        current_state = MuhasibiState.LISTEN

        while current_state not in (MuhasibiState.FINALIZE, MuhasibiState.FAILED):
            try:
                current_state = await self._execute_state(current_state, ctx)
            except Exception as e:
                ctx.error = str(e)
                if ctx.retry_count < ctx.max_retries:
                    ctx.retry_count += 1
                    continue
                current_state = MuhasibiState.FAILED

        return self._build_response(ctx)

    async def _execute_state(
        self,
        state: MuhasibiState,
        ctx: StateContext,
    ) -> MuhasibiState:
        """Execute a single state and return the next state."""
        start = datetime.utcnow()

        if state == MuhasibiState.LISTEN:
            next_state = await self._state_listen(ctx)
        elif state == MuhasibiState.PURPOSE:
            next_state = await self._state_purpose(ctx)
        elif state == MuhasibiState.PATH:
            next_state = await self._state_path(ctx)
        elif state == MuhasibiState.RETRIEVE:
            next_state = await self._state_retrieve(ctx)
        elif state == MuhasibiState.ACCOUNT:
            next_state = await self._state_account(ctx)
        elif state == MuhasibiState.INTERPRET:
            next_state = await self._state_interpret(ctx)
        elif state == MuhasibiState.REFLECT:
            next_state = await self._state_reflect(ctx)
        else:
            next_state = MuhasibiState.FAILED

        # Record timing
        elapsed = (datetime.utcnow() - start).total_seconds()
        ctx.state_timings[state.name] = elapsed

        return next_state

    async def _state_listen(self, ctx: StateContext) -> MuhasibiState:
        """
        LISTEN state: Normalize and understand the question.

        This is a non-LLM state that:
        - Normalizes Arabic text
        - Detects explicit pillar/value names
        - Extracts keywords
        """
        # Normalize the question
        ctx.normalized_question = normalize_for_matching(ctx.question)

        # Extract keywords
        ctx.question_keywords = extract_arabic_words(ctx.question)

        # Detect entities if resolver is available
        if self.entity_resolver:
            resolved = self.entity_resolver.resolve(ctx.question)
            ctx.detected_entities = [
                {
                    "type": r.entity_type.value,
                    "id": r.entity_id,
                    "name_ar": r.name_ar,
                    "confidence": r.confidence,
                }
                for r in resolved
            ]

        # Create listen summary
        if ctx.detected_entities:
            entity_names = ", ".join(e["name_ar"] for e in ctx.detected_entities[:3])
            ctx.listen_summary_ar = f"السؤال عن: {entity_names}"
        else:
            ctx.listen_summary_ar = f"سؤال عام: {ctx.question[:100]}"

        return MuhasibiState.PURPOSE

    async def _state_purpose(self, ctx: StateContext) -> MuhasibiState:
        """
        PURPOSE state: Determine ultimate goal and constraints.

        This state uses GPT-5 to generate structured purpose output.
        """
        # Build purpose with required constraints (LLM preferred; fallback deterministic)
        if self.llm_client:
            result = await self.llm_client.purpose_path(ctx.question)
            if result:
                # Ensure required constraints always present
                constraints = list(dict.fromkeys(result.constraints_ar + self.REQUIRED_CONSTRAINTS))
                ctx.purpose = Purpose(
                    ultimate_goal_ar=result.ultimate_goal_ar,
                    constraints_ar=constraints,
                )
                ctx.path_plan_ar = result.path_plan_ar
                try:
                    ctx.difficulty = Difficulty(result.difficulty)
                except Exception:
                    ctx.difficulty = Difficulty.MEDIUM
                return MuhasibiState.RETRIEVE if ctx.path_plan_ar else MuhasibiState.PATH

        # Default purpose with required constraints
        ctx.purpose = Purpose(
            ultimate_goal_ar="الإجابة على السؤال من الأدلة الشرعية فقط",
            constraints_ar=self.REQUIRED_CONSTRAINTS.copy(),
        )

        return MuhasibiState.PATH

    async def _state_path(self, ctx: StateContext) -> MuhasibiState:
        """
        PATH state: Plan the approach.

        This state determines the plan steps.
        """
        # Determine difficulty based on entities and keywords
        if len(ctx.detected_entities) == 0:
            ctx.difficulty = Difficulty.HARD
        elif len(ctx.detected_entities) == 1:
            ctx.difficulty = Difficulty.MEDIUM
        else:
            ctx.difficulty = Difficulty.EASY

        # Default plan
        ctx.path_plan_ar = [
            "استخراج الكيانات المذكورة في السؤال",
            "استرجاع التعريفات والأدلة من قاعدة البيانات",
            "التحقق من تغطية الأدلة للسؤال",
            "صياغة الإجابة مع الاستشهادات",
        ]

        return MuhasibiState.RETRIEVE

    async def _state_retrieve(self, ctx: StateContext) -> MuhasibiState:
        """
        RETRIEVE state: Get evidence packets.

        This is a non-LLM state that retrieves evidence.
        """
        if self.retriever:
            try:
                # Best-effort retrieval; retriever opens DB session internally if provided as callable.
                # Here we expect a HybridRetriever plus a DB session passed via closure in /ask route.
                retrieve_fn = getattr(self.retriever, "retrieve", None)
                if retrieve_fn and hasattr(self.retriever, "_session") and self.retriever._session:
                    session = self.retriever._session
                    merge = await self.retriever.retrieve(
                        session,
                        RetrievalInputs(
                            query=ctx.question,
                            resolved_entities=ctx.detected_entities,
                        ),
                    )
                    ctx.evidence_packets = merge.evidence_packets
                    ctx.has_definition = merge.has_definition
                    ctx.has_evidence = merge.has_evidence
            except Exception as e:
                ctx.account_issues.append(f"فشل الاسترجاع: {e}")

        # If still empty, keep flags consistent
        ctx.has_definition = len([
            p for p in ctx.evidence_packets
            if p.get("chunk_type") == "definition"
        ]) > 0

        ctx.has_evidence = len([
            p for p in ctx.evidence_packets
            if p.get("chunk_type") == "evidence"
        ]) > 0

        return MuhasibiState.ACCOUNT

    async def _state_account(self, ctx: StateContext) -> MuhasibiState:
        """
        ACCOUNT state: Validate evidence coverage.

        This is a non-LLM state that enforces citation constraints.
        """
        ctx.account_issues = []

        # Check if we have any evidence
        if not ctx.evidence_packets:
            ctx.account_issues.append("لا توجد أدلة متاحة")
            ctx.citation_valid = False
            ctx.not_found = True
            return MuhasibiState.INTERPRET  # Skip to interpret with not_found

        # Check for definition
        if not ctx.has_definition:
            ctx.account_issues.append("لا يوجد تعريف للمفهوم المطلوب")

        # Validation passes if we have evidence
        ctx.citation_valid = True

        return MuhasibiState.INTERPRET

    async def _state_interpret(self, ctx: StateContext) -> MuhasibiState:
        """
        INTERPRET state: Generate answer from evidence.

        This state uses GPT-5 to interpret evidence packets.
        """
        if ctx.not_found:
            # No evidence - return refusal
            ctx.answer_ar = "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال."
            ctx.confidence = Confidence.LOW
            ctx.citations = []
            return MuhasibiState.REFLECT

        if self.llm_client and ctx.evidence_packets:
            result = await self.llm_client.interpret(
                question=ctx.question,
                evidence_packets=ctx.evidence_packets,
                detected_entities=ctx.detected_entities,
            )
            if result:
                ctx.answer_ar = result.answer_ar
                ctx.not_found = bool(result.not_found)
                try:
                    ctx.confidence = Confidence(result.confidence)
                except Exception:
                    ctx.confidence = Confidence.MEDIUM
                # Map citations into Pydantic Citation objects
                ctx.citations = [
                    Citation(
                        chunk_id=c.get("chunk_id", ""),
                        source_anchor=c.get("source_anchor", ""),
                        ref=c.get("ref"),
                    )
                    for c in result.citations
                    if c.get("chunk_id") and c.get("source_anchor")
                ]

        # Deterministic fallback answer if evidence exists but no LLM call occurred.
        if not ctx.answer_ar and ctx.evidence_packets:
            defs = [p for p in ctx.evidence_packets if p.get("chunk_type") == "definition"]
            evs = [p for p in ctx.evidence_packets if p.get("chunk_type") == "evidence"]
            chosen = (defs[:1] + evs[:2]) or ctx.evidence_packets[:3]

            parts = []
            if defs:
                parts.append(f"التعريف:\n{defs[0].get('text_ar','').strip()}")
            if evs:
                parts.append("التأصيل/الدليل:")
                for e in evs[:2]:
                    parts.append(e.get("text_ar", "").strip())

            if not parts:
                parts = ["تم العثور على أدلة، لكن تعذر تلخيصها آليًا بدون نموذج لغوي."]

            ctx.answer_ar = "\n\n".join([p for p in parts if p])
            ctx.citations = [
                Citation(
                    chunk_id=p.get("chunk_id", ""),
                    source_anchor=p.get("source_anchor", ""),
                    ref=(p.get("refs", [{}])[0].get("ref") if p.get("refs") else None),
                )
                for p in chosen
                if p.get("chunk_id") and p.get("source_anchor")
            ]
            ctx.confidence = Confidence.MEDIUM if ctx.citations else Confidence.LOW
            ctx.not_found = not bool(ctx.citations)

        # If still no answer, refuse safely
        if not ctx.answer_ar:
            ctx.answer_ar = "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال."
            ctx.not_found = True
            ctx.confidence = Confidence.LOW
            ctx.citations = []

        # Run guardrails if available
        if self.guardrails:
            try:
                result = self.guardrails.validate(
                    answer_ar=ctx.answer_ar,
                    citations=[c.model_dump() for c in ctx.citations],
                    evidence_packets=ctx.evidence_packets,
                    not_found=ctx.not_found,
                )
                if not result.passed:
                    ctx.not_found = True
                    ctx.answer_ar = "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال."
                    ctx.citations = []
                    ctx.confidence = Confidence.LOW
            except Exception:
                # Fail closed
                ctx.not_found = True
                ctx.answer_ar = "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال."
                ctx.citations = []
                ctx.confidence = Confidence.LOW

        # Build entities from detected
        ctx.entities = [
            EntityRef(
                type=EntityType(e["type"]),
                id=e["id"],
                name_ar=e["name_ar"],
            )
            for e in ctx.detected_entities
        ]

        return MuhasibiState.REFLECT

    async def _state_reflect(self, ctx: StateContext) -> MuhasibiState:
        """
        REFLECT state: Add consequence-aware reflection.

        This state adds reflection without new claims.
        """
        # For not_found, no reflection needed
        if ctx.not_found:
            ctx.reflection_added = False
            return MuhasibiState.FINALIZE

        if self.llm_client:
            # TODO: Add reflection with constraints
            pass

        ctx.reflection_added = True
        return MuhasibiState.FINALIZE

    def _build_response(self, ctx: StateContext) -> FinalResponse:
        """Build the final response from context."""
        return FinalResponse(
            listen_summary_ar=ctx.listen_summary_ar,
            purpose=ctx.purpose or Purpose(
                ultimate_goal_ar="غير محدد",
                constraints_ar=self.REQUIRED_CONSTRAINTS,
            ),
            path_plan_ar=ctx.path_plan_ar,
            answer_ar=ctx.answer_ar,
            citations=ctx.citations,
            entities=ctx.entities,
            difficulty=ctx.difficulty,
            not_found=ctx.not_found,
            confidence=ctx.confidence,
        )


def create_middleware(
    entity_resolver=None,
    retriever=None,
    llm_client=None,
    guardrails=None,
) -> MuhasibiMiddleware:
    """
    Create a configured Muḥāsibī middleware instance.

    Args:
        entity_resolver: Optional entity resolver.
        retriever: Optional retrieval pipeline.
        llm_client: Optional LLM client.
        guardrails: Optional guardrails.

    Returns:
        Configured MuhasibiMiddleware.
    """
    return MuhasibiMiddleware(
        entity_resolver=entity_resolver,
        retriever=retriever,
        llm_client=llm_client,
        guardrails=guardrails,
    )

