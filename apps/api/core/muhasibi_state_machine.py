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
from apps.api.core.contract_gate import apply_runtime_contract_gate
from apps.api.core.muhasibi_trace import summarize_state
from apps.api.core.muhasibi_account import apply_question_evidence_relevance_gate
from apps.api.core.muhasibi_listen import run_listen
from apps.api.core.muhasibi_reasoning import ChainStage, build_reasoning_trace


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
    mode: str = "natural_chat"  # natural_chat|answer|debate|socratic|judge

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
        """Execute a single state and return the next state.

        Root-cause guard:
        - This pipeline must never hang indefinitely in production.
        - We add per-state timeouts + start/end logging so we can pinpoint the exact blocker.
        """
        import asyncio
        import hashlib
        import logging

        logger = logging.getLogger(__name__)
        qid = hashlib.sha1((ctx.question or "").encode("utf-8")).hexdigest()[:8]
        start = datetime.utcnow()

        # Deterministic per-state timeouts (seconds).
        # Reason: prevent request-level deadlocks and make issues observable.
        timeouts: dict[MuhasibiState, float] = {
            MuhasibiState.LISTEN: 15.0,
            MuhasibiState.PURPOSE: 90.0,
            MuhasibiState.PATH: 15.0,
            MuhasibiState.RETRIEVE: 90.0,
            MuhasibiState.ACCOUNT: 45.0,
            MuhasibiState.INTERPRET: 180.0,
            MuhasibiState.REFLECT: 90.0,
        }

        async def _run_state() -> MuhasibiState:
            if state == MuhasibiState.LISTEN:
                return await self._state_listen(ctx)
            if state == MuhasibiState.PURPOSE:
                return await self._state_purpose(ctx)
            if state == MuhasibiState.PATH:
                return await self._state_path(ctx)
            if state == MuhasibiState.RETRIEVE:
                return await self._state_retrieve(ctx)
            if state == MuhasibiState.ACCOUNT:
                return await self._state_account(ctx)
            if state == MuhasibiState.INTERPRET:
                return await self._state_interpret(ctx)
            if state == MuhasibiState.REFLECT:
                return await self._state_reflect(ctx)
            return MuhasibiState.FAILED

        timeout_s = float(timeouts.get(state, 120.0))
        logger.info(f"[MUHASIBI] qid={qid} state_start={state.name} timeout_s={timeout_s}")
        try:
            next_state = await asyncio.wait_for(_run_state(), timeout=timeout_s)
        except asyncio.TimeoutError:
            ctx.error = f"Timeout in state {state.name} after {timeout_s:.0f}s"
            logger.error(f"[MUHASIBI] qid={qid} state_timeout={state.name} after_s={timeout_s}")
            return MuhasibiState.FAILED
        except Exception as e:
            # Preserve existing retry behavior in the outer loop; still log here for diagnosis.
            logger.exception(f"[MUHASIBI] qid={qid} state_error={state.name} err={e}")
            raise

        # Record timing
        elapsed = (datetime.utcnow() - start).total_seconds()
        ctx.state_timings[state.name] = elapsed
        if getattr(ctx, "trace_enabled", False):
            snap = summarize_state(state.name, ctx)
            snap["elapsed_s"] = elapsed
            ctx.trace.append(snap)

        logger.info(f"[MUHASIBI] qid={qid} state_done={state.name} elapsed_s={elapsed:.3f}")
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
        # Extract intent_type at method level for reranker policy decisions
        intent_type = None
        if hasattr(ctx, "intent") and ctx.intent:
            intent_type = ctx.intent.get("intent_type")
        
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
                            intent=intent_type,
                        ),
                    )
                    ctx.evidence_packets = merge.evidence_packets
                    ctx.has_definition = merge.has_definition
                    ctx.has_evidence = merge.has_evidence
            except Exception as e:
                ctx.account_issues.append(f"فشل الاسترجاع: {e}")

        # Deterministic retrieval hints from Muḥāsibī chain stage (no LLM).
        # Reason: action/habit questions benefit from procedural phrasing and can improve recall.
        try:
            if (
                self.retriever
                and hasattr(self.retriever, "_session")
                and self.retriever._session
                and getattr(self.retriever, "retrieve", None)
                and ctx.detected_entities
            ):
                trace = build_reasoning_trace(
                    question=ctx.question,
                    detected_entities=ctx.detected_entities,
                    evidence_packets=ctx.evidence_packets,
                    intent=ctx.intent,
                    difficulty=getattr(ctx.difficulty, "value", None),
                )
                if trace.chain_stage in {ChainStage.ACTION, ChainStage.HABIT, ChainStage.FIXED_INTENTION}:
                    session = self.retriever._session
                    extra_packets: list[dict[str, Any]] = []
                    names = [str(e.get("name_ar") or "") for e in (ctx.detected_entities or []) if e.get("name_ar")]
                    hint_queries: list[str] = []
                    for n in names[:2]:
                        hint_queries.extend([f"{n} إجرائي", f"{n} خطوات", f"كيفية تطبيق {n}"])
                    for q in hint_queries[:4]:
                        try:
                            merge2 = await self.retriever.retrieve(
                                session,
                                RetrievalInputs(query=q, resolved_entities=ctx.detected_entities, intent=intent_type),
                            )
                            extra_packets.extend(merge2.evidence_packets)
                        except Exception:
                            continue

                    # Merge deterministically: keep existing first, then add unique chunk_ids.
                    if extra_packets:
                        seen = {p.get("chunk_id") for p in ctx.evidence_packets if p.get("chunk_id")}
                        for p in extra_packets:
                            cid = p.get("chunk_id")
                            if cid and cid not in seen:
                                ctx.evidence_packets.append(p)
                                seen.add(cid)
                        ctx.evidence_packets = ctx.evidence_packets[:12]
        except Exception:
            pass

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
                                    intent=intent_type,
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

        # SEED RETRIEVAL FLOOR for broad intents
        # Reason: global_synthesis, cross_pillar, guidance_framework_chat, and natural_chat
        # must never abstain with 0 citations due to missing entity anchors.
        # Always include pillar definitions + bridge notes for these intents.
        try:
            intent = getattr(ctx, "intent", None) or {}
            intent_type = str(intent.get("intent_type") or "")
            mode = getattr(ctx, "mode", "") or ""
            
            needs_seed_floor = (
                intent_type in {
                    "system_limits_policy",
                    "guidance_framework_chat",
                    "global_synthesis",
                    "cross_pillar",
                    "network_build",
                    "compare",
                    "world_model",
                }
                or intent.get("requires_seed_retrieval")
                or intent.get("bypass_relevance_gate")
                or (mode == "natural_chat" and not ctx.detected_entities)
            )
            
            if needs_seed_floor and self.retriever and hasattr(self.retriever, "_session") and self.retriever._session:
                session = self.retriever._session
                await self._add_seed_retrieval_floor(ctx, session)
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

    async def _add_seed_retrieval_floor(self, ctx: StateContext, session) -> None:
        """
        Add seed evidence floor for broad/synthesis intents.
        
        Uses cached, deterministic seed bundles to prevent:
        - Nondeterminism from varying DB query results
        - "Warmup variance" causing false abstentions
        
        Seeds (cached on first load):
        - 5 pillar definition chunks (one per pillar, deterministic order)
        - Top 10 cross-pillar edges (deterministic order by edge_id)
        - Policy packet for system_limits_policy intent
        """
        from apps.api.retrieve.seed_cache import (
            SeedCache, load_global_seed_bundle, get_policy_packet
        )
        import logging
        
        logger = logging.getLogger(__name__)
        cache = SeedCache.get_instance()
        
        # Get or load the global seed bundle (cached after first load)
        bundle = cache.get_global_bundle()
        if bundle is None:
            try:
                bundle = await load_global_seed_bundle(session)
                cache.set_global_bundle(bundle)
                logger.debug("Loaded and cached global seed bundle")
            except Exception as e:
                logger.warning(f"Failed to load seed bundle: {e}")
                bundle = None
        
        if bundle is None or bundle.is_empty:
            logger.warning("Seed bundle is empty or failed to load")
            return
        
        # Add seed packets (deterministic order from cache)
        existing_chunk_ids = {p.get("chunk_id") for p in ctx.evidence_packets if p.get("chunk_id")}
        
        for packet in bundle.all_packets:
            cid = packet.get("chunk_id")
            if cid and cid not in existing_chunk_ids:
                ctx.evidence_packets.append(packet.copy())
                existing_chunk_ids.add(cid)
        
        # For system_limits_policy, add policy response packet
        try:
            intent = getattr(ctx, "intent", None) or {}
            intent_type_val = str(intent.get("intent_type") or "")
            if intent_type_val == "system_limits_policy":
                policy_packet = get_policy_packet()
                if policy_packet["chunk_id"] not in existing_chunk_ids:
                    ctx.evidence_packets.insert(0, policy_packet)
        except Exception:
            pass

    async def _state_account(self, ctx: StateContext) -> MuhasibiState:
        """
        ACCOUNT state: Validate evidence coverage.

        This is a non-LLM state that enforces citation constraints.
        """
        ctx.account_issues = []

        # Check if we have any evidence
        if not ctx.evidence_packets:
            # For synthesis intents, don't set not_found even if empty
            # Reason: synthesis questions can sometimes get 0 evidence due to
            # flaky retrieval, but the bypass in apply_question_evidence_relevance_gate
            # should prevent false abstention. Also, seed floor should have added seeds.
            intent = getattr(ctx, "intent", None) or {}
            intent_type = str(intent.get("intent_type") or "")
            synthesis_intents = {
                "global_synthesis", "cross_pillar", "network_build",
                "compare", "world_model", "guidance_framework_chat",
            }
            
            # SAFEGUARD: If truly catastrophic (0 packets AND seed cache empty),
            # still abstain even for synthesis intents. This prevents weird outputs
            # during DB outages or cache failures.
            from apps.api.retrieve.seed_cache import SeedCache
            seed_cache = SeedCache.get_instance()
            seed_bundle = seed_cache.get_global_bundle()
            catastrophic_failure = (
                seed_bundle is None or seed_bundle.is_empty
            )
            
            if intent_type not in synthesis_intents or catastrophic_failure:
                ctx.account_issues.append("لا توجد أدلة متاحة")
                ctx.citation_valid = False
                ctx.not_found = True
                if catastrophic_failure and intent_type in synthesis_intents:
                    ctx.account_issues.append("فشل تحميل البذور الأساسية - يُرجى المحاولة لاحقًا")
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
        # Runtime contract gate: enforce intent/coverage before FINALIZE.
        # Reason: production must not emit safe-but-off-target answers.
        await apply_runtime_contract_gate(middleware=self, ctx=ctx, enable_repair=True)
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

