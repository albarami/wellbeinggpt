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
from apps.api.core.muhasibi_interpret import run_interpret, run_reflect, build_final_response
from apps.api.core.muhasibi_trace import summarize_state
from apps.api.core.muhasibi_account import apply_question_evidence_relevance_gate
from apps.api.core.muhasibi_listen import run_listen


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
    mode: str = "answer"  # answer|debate|socratic|judge

    # LISTEN outputs
    normalized_question: str = ""
    listen_summary_ar: str = ""
    detected_entities: list[dict] = field(default_factory=list)
    question_keywords: list[str] = field(default_factory=list)
    intent: Optional[dict[str, Any]] = None

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
    refusal_suggestion_ar: Optional[str] = None

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
    trace_enabled: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)


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

    async def process(self, question: str, language: str = "ar", mode: str = "answer") -> FinalResponse:
        """
        Process a question through the full Muḥāsibī pipeline.

        Args:
            question: The user's question.
            language: Language preference.

        Returns:
            FinalResponse following the required schema.
        """
        ctx = StateContext(question=question, language=language, mode=mode)

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

    async def process_with_trace(
        self,
        question: str,
        language: str = "ar",
        mode: str = "answer",
    ) -> tuple[FinalResponse, list[dict[str, Any]]]:
        """
        Process a question and return a safe Muḥāsibī trace.
        """
        ctx = StateContext(question=question, language=language, mode=mode, trace_enabled=True)

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

        return self._build_response(ctx), ctx.trace

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
        if getattr(ctx, "trace_enabled", False):
            snap = summarize_state(state.name, ctx)
            snap["elapsed_s"] = elapsed
            ctx.trace.append(snap)

        return next_state

    async def _state_listen(self, ctx: StateContext) -> MuhasibiState:
        """
        LISTEN state: Normalize and understand the question.

        This is a non-LLM state that:
        - Normalizes Arabic text
        - Detects explicit pillar/value names
        - Extracts keywords
        """
        await run_listen(self, ctx)

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

        # If retrieval is weak, use GPT-5 ONLY to rewrite the query (not to answer),
        # then run additional non-LLM retrieval attempts.
        if (not ctx.evidence_packets or len(ctx.evidence_packets) < 3) and self.llm_client and ctx.language == "ar":
            try:
                rewrites = await self.llm_client.query_rewrite_ar(
                    question=ctx.question,
                    detected_entities=ctx.detected_entities,
                    keywords=ctx.question_keywords,
                )
                if rewrites and self.retriever and hasattr(self.retriever, "_session") and self.retriever._session:
                    session = self.retriever._session
                    extra_packets: list[dict[str, Any]] = []
                    for q in list(rewrites.get("rewrites_ar", []))[:3]:
                        try:
                            merge = await self.retriever.retrieve(
                                session,
                                RetrievalInputs(
                                    query=str(q),
                                    resolved_entities=ctx.detected_entities,
                                ),
                            )
                            extra_packets.extend(merge.evidence_packets)
                        except Exception:
                            continue

                    # Merge deterministically: keep existing first, then add new unique chunk_ids.
                    seen = {p.get("chunk_id") for p in ctx.evidence_packets if p.get("chunk_id")}
                    for p in extra_packets:
                        cid = p.get("chunk_id")
                        if cid and cid not in seen:
                            ctx.evidence_packets.append(p)
                            seen.add(cid)

                    ctx.evidence_packets = ctx.evidence_packets[:12]
            except Exception:
                pass

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

        # Out-of-scope gate: if evidence isn't relevant to the question, fail closed.
        apply_question_evidence_relevance_gate(ctx)

        # Validation passes if we have evidence
        if not ctx.not_found:
            ctx.citation_valid = True

        return MuhasibiState.INTERPRET

    async def _state_interpret(self, ctx: StateContext) -> MuhasibiState:
        """
        INTERPRET state: Generate answer from evidence.

        This state uses GPT-5 to interpret evidence packets.
        """
        await run_interpret(self, ctx)

        return MuhasibiState.REFLECT

    async def _state_reflect(self, ctx: StateContext) -> MuhasibiState:
        """
        REFLECT state: Add consequence-aware reflection.

        This state adds reflection without new claims.
        """
        await run_reflect(self, ctx)
        return MuhasibiState.FINALIZE

    def _build_response(self, ctx: StateContext) -> FinalResponse:
        """Build the final response from context."""
        return build_final_response(self, ctx)


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

