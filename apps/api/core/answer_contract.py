"""Answer contract enforcement (shared by runtime + eval).

Goal:
- Enforce *intent satisfaction* deterministically while preserving evidence-only safety.
- Provide a single contract checker used in:
  - runtime (Muḥāsibī INTERPRET)
  - evaluation runner + reporting

This does NOT add new knowledge. It only checks required structure/coverage and
decides whether to repair / partially answer / abstain.

Reason: stakeholder acceptance showed \"safe but off-target\" failure modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from apps.api.retrieve.normalize_ar import normalize_for_matching


@dataclass(frozen=True)
class ContractOutcome:
    """
    Contract evaluation result.

    - PASS_FULL: requirements satisfied (including graph when required)
    - PASS_PARTIAL: produced an honest, grounded partial (A+B) when full cannot be supported
    - FAIL: contract unmet
    """

    value: str  # PASS_FULL|PASS_PARTIAL|FAIL


@dataclass(frozen=True)
class UsedEdgeSpan:
    chunk_id: str
    span_start: int
    span_end: int
    quote: str


@dataclass(frozen=True)
class UsedEdge:
    """An edge the answer *actually relied on* (not merely retrieved)."""

    edge_id: str
    from_node: str  # e.g. "pillar:P004"
    to_node: str  # e.g. "pillar:P001"
    relation_type: str  # semantic enum
    justification_spans: tuple[UsedEdgeSpan, ...]


@dataclass(frozen=True)
class ArgumentChain:
    """
    Deterministic, inspectable argument chain (no CoT).

    This is built from a UsedEdge and its justification spans.
    """

    edge_id: str
    relation_type: str
    from_node: str
    to_node: str
    claim_ar: str
    inference_type: str  # direct_quote | multi_span_entailment
    evidence_spans: tuple[UsedEdgeSpan, ...]
    boundary_ar: str
    boundary_spans: tuple[UsedEdgeSpan, ...]


def _is_boundary_text(text_ar: str) -> bool:
    t = normalize_for_matching(text_ar or "")
    if not t:
        return False
    markers = [
        "ضوابط",
        "حدود",
        "ميزان",
        "انحراف",
        "افراط",
        "تفريط",
        "لا ينبغي",
        "لا يجوز",
        "لا يصح",
        "تحذير",
        "تنبيه",
        "محاذير",
        "لا يتحقق",
        "متوقف على",
        "مشروط",
        "شرط",
    ]
    return any(normalize_for_matching(m) in t for m in markers)


def _relation_ar(rt: str) -> str:
    rt = str(rt or "").strip().upper()
    return {
        "ENABLES": "تمكين/إعانة",
        "REINFORCES": "تعزيز",
        "COMPLEMENTS": "تكامل",
        "CONDITIONAL_ON": "مشروط بـ",
        "TENSION_WITH": "تعارض ظاهري",
        "RESOLVES_WITH": "توفيق/ترجيح",
    }.get(rt, rt or "علاقة")


def build_argument_chains_from_used_edges(*, used_edges: list[UsedEdge]) -> list[ArgumentChain]:
    """
    Build deterministic argument chains from used edges.

    Rules:
    - Evidence spans: take up to 3 non-boundary spans; if none, take first span.
    - Boundary: if any span looks like a boundary, use the first one; else "غير منصوص عليه في الإطار".
    - Inference type: direct_quote when 1 evidence span, otherwise multi_span_entailment.
    """

    out: list[ArgumentChain] = []
    for ue in (used_edges or [])[:24]:
        spans = list(ue.justification_spans or [])
        if not spans:
            continue

        boundary_spans = [sp for sp in spans if _is_boundary_text(sp.quote)]
        ev_spans = [sp for sp in spans if sp not in boundary_spans]
        ev_pick = tuple(ev_spans[:3] or spans[:1])
        b_pick = tuple(boundary_spans[:2])
        boundary_ar = str(b_pick[0].quote).strip() if b_pick else "غير منصوص عليه في الإطار"

        inference_type = "direct_quote" if len(ev_pick) == 1 else "multi_span_entailment"
        claim_ar = f"({ _relation_ar(ue.relation_type) }) {ue.from_node} → {ue.to_node}"

        out.append(
            ArgumentChain(
                edge_id=str(ue.edge_id or ""),
                relation_type=str(ue.relation_type or ""),
                from_node=str(ue.from_node or ""),
                to_node=str(ue.to_node or ""),
                claim_ar=claim_ar,
                inference_type=inference_type,
                evidence_spans=ev_pick,
                boundary_ar=boundary_ar,
                boundary_spans=b_pick,
            )
        )
    return out


@dataclass(frozen=True)
class ContractSpec:
    """Deterministic intent/coverage requirements."""

    intent_type: str  # compare|cross_pillar|network|tension|scenario|generic
    required_sections: tuple[str, ...] = ()
    required_entities: tuple[str, ...] = ()  # Arabic surface forms to mention

    # Graph requirements
    requires_graph: bool = False
    min_links: int = 0
    min_distinct_pillars: int = 0

    # Stakeholder acceptance: avoid \"أسئلة توضيح\" unless explicitly requested.
    allow_followup_questions: bool = False


@dataclass(frozen=True)
class ContractMetrics:
    outcome: ContractOutcome
    reasons: tuple[str, ...]

    section_nonempty: float
    required_entities_coverage: float
    graph_required_satisfied: bool


def extract_compare_concepts_from_question(question_ar: str) -> tuple[str, ...]:
    """
    Question-agnostic extraction of compare concepts.
    
    Handles multiple patterns:
    1. Quoted concepts: "الإيمان" و"العبادة" و"التزكية"
    2. وبين pattern: الفرق بينها وبين X و Y
    3. بين X و Y pattern
    """
    import re
    
    q = (question_ar or "").strip()
    if not q:
        return ()
    
    out: list[str] = []
    
    # Pattern 1: Extract quoted concepts (most reliable)
    # Handles "concept", «concept», and "concept" (Arabic/smart quotes)
    quoted = re.findall(r'[""«]([^""»]+)[""»]', q)
    if len(quoted) >= 2:
        for concept in quoted[:6]:
            c = concept.strip()
            if 0 < len(c) <= 30:
                out.append(c)
        if out:
            return tuple(out)
    
    # Pattern 2: "وبين" pattern - "الفرق بينها وبين X و Y ..."
    if "وبين" in q:
        _, _, tail = q.partition("وبين")
        tail = tail.strip()
        for stop in ["من حيث", ":", "؛", ".", "؟", "\n", "(", "（", "داخل"]:
            if stop in tail:
                tail = tail.split(stop, 1)[0].strip()
        
        parts = [p.strip(" ،,\"'«»""") for p in tail.split("و") if p.strip(" ،,\"'«»""")]
        for p in parts[:4]:
            if 0 < len(p) <= 24:
                out.append(p)
        if out:
            return tuple(out)
    
    # Pattern 3: "بين X و Y" pattern (after فرق/قارن/مقارنة)
    match = re.search(r'(?:فر[قّ]|قارن|مقارنة)\s+بين\s+(.+?)(?:من حيث|داخل|:|؟|$)', q)
    if match:
        segment = match.group(1).strip()
        # Split by "و" 
        parts = [p.strip(" ،,\"'«»""") for p in segment.split("و") if p.strip(" ،,\"'«»""")]
        for p in parts[:4]:
            if 0 < len(p) <= 24:
                out.append(p)
    
    return tuple(out)


def contract_from_answer_requirements(
    *,
    question_norm: str,
    question_ar: str,
    question_type: str,
    answer_requirements: dict[str, Any],
) -> ContractSpec:
    """
    Build a ContractSpec from dataset `answer_requirements`.

    This is the eval-side entrypoint so eval and runtime share the same checker.
    """

    must_include = list((answer_requirements or {}).get("must_include") or [])
    fmt = str((answer_requirements or {}).get("format") or "").strip()
    intent_type = str(question_type or "generic").strip() or "generic"
    if intent_type == "contradiction":
        intent_type = "tension"

    # Some datasets (e.g. stakeholder acceptance) use a non-intent `type` label.
    # In that case, infer intent deterministically from the question text.
    allowed_intents = {"compare", "scenario", "cross_pillar", "network", "tension", "cross_pillar_path", "generic"}
    if intent_type not in allowed_intents:
        q = question_norm or normalize_for_matching(question_ar or "")
        if any(k in q for k in ["قارن", "المقارنة", "ما الفرق", "الفرق", "بيّن الفرق"]):
            intent_type = "compare"
        elif any(k in q for k in ["شبكة", "ابن شبكة", "اربطها بثلاث", "ثلاث ركائز"]):
            intent_type = "network"
        elif any(k in q for k in ["تعارض", "التعارض", "توفيق", "ترجيح", "كيف نجمع"]):
            intent_type = "tension"
        elif any(k in q for k in ["حلل", "حلّل", "سلسلة التدخل", "علامة قياس", "مؤشر", "حالة", "سيناريو"]):
            intent_type = "scenario"
        elif any(k in q for k in ["مسار", "خطوة بخطوة", "قدّم مسار", "خطوات الربط"]):
            intent_type = "cross_pillar_path"
        elif any(k in q for k in ["ما العلاقة", "العلاقة بين", "اربط بين", "ربط بين"]):
            intent_type = "cross_pillar"
        else:
            intent_type = "generic"

    required_sections: list[str] = []
    # The dataset uses "scholar" format but type controls the contract.
    if intent_type == "compare":
        required_sections = ["مصفوفة المقارنة", "خلاصة تنفيذية (3 نقاط)"]
    elif intent_type == "scenario":
        required_sections = [
            # Naturalized partial headings (avoid A/B labels)
            "ما يمكن دعمه من الأدلة المسترجعة",
            "ما لا يمكن الجزم به من الأدلة الحالية",
        ]
    else:
        # Default scholar sections (subset-based on must_include)
        for h in [
            "تعريف المفهوم داخل الإطار",
            "التأصيل والأدلة",
            "التأصيل والأدلة (مختصر ومركز)",
            "الربط بين الركائز",
            "الربط بين الركائز (مع سبب الربط)",
            "تطبيق عملي",
            "تطبيق عملي على الحالة/السؤال",
            "تنبيهات وأخطاء شائعة",
            "خلاصة تنفيذية",
            "خلاصة تنفيذية (3 نقاط)",
        ]:
            if any(h in str(x) for x in must_include):
                # Normalize older headings to the strict ones our composer emits.
                if h == "التأصيل والأدلة":
                    h = "التأصيل والأدلة (مختصر ومركز)"
                if h == "الربط بين الركائز":
                    h = "الربط بين الركائز (مع سبب الربط)"
                if h == "تطبيق عملي":
                    h = "تطبيق عملي على الحالة/السؤال"
                if h == "خلاصة تنفيذية":
                    h = "خلاصة تنفيذية (3 نقاط)"
                if h not in required_sections:
                    required_sections.append(h)

    requires_graph = (
        bool((answer_requirements or {}).get("path_trace"))
        or any("path_trace" in str(x) for x in must_include)
        or (intent_type in {"cross_pillar", "network", "tension"})
    )

    required_entities: tuple[str, ...] = ()
    if intent_type == "compare":
        # Prefer deterministic extraction from the question itself.
        concepts = extract_compare_concepts_from_question(question_ar)
        required_entities = tuple([c for c in concepts if c])

    return ContractSpec(
        intent_type=intent_type,
        required_sections=tuple(required_sections),
        required_entities=required_entities,
        requires_graph=requires_graph,
        min_links=3 if (intent_type == "network") else (1 if requires_graph else 0),
        min_distinct_pillars=4 if (intent_type == "network") else 0,
        allow_followup_questions=False,
    )


def contract_from_question_runtime(
    *,
    question_norm: str,
    detected_entities: list[dict[str, Any]],
) -> ContractSpec:
    """
    Runtime contract inference (heuristics).

    Reason: production requests don’t have dataset metadata.
    """

    q = question_norm or ""

    intent_type = "generic"
    
    # System limits / policy questions: PASS_FULL without graph requirements
    # These are answerable from system policy (not framework evidence)
    policy_markers = ["حدود الربط", "ما حدود", "حدود النظام", "غير المنصوص", "غير منصوص"]
    if any(m in q for m in policy_markers):
        intent_type = "system_limits_policy"
    elif any(k in q for k in ["قارن", "المقارنة", "الفرق", "ما الفرق"]):
        intent_type = "compare"
    elif any(k in q for k in ["شبكة", "ابن", "ثلاث ركائز", "اربطها بثلاث"]):
        intent_type = "network"
    elif any(k in q for k in ["تعارض", "التعارض", "توفيق", "ترجيح", "كيف نجمع"]):
        intent_type = "tension"
    elif any(k in q for k in ["حلل", "حلّل", "سلسلة التدخل", "مؤشر", "علامة قياس"]):
        intent_type = "scenario"
    elif any(k in q for k in ["ما العلاقة", "العلاقة بين", "اربط بين", "ربط بين", "مسار", "خطوة بخطوة"]):
        intent_type = "cross_pillar"

    requires_graph = intent_type in {"cross_pillar", "network", "tension"}

    # Required Arabic surface forms:
    # - Only enforce entity mention coverage for non-generic intents to avoid
    #   regressions on simple structure/list questions.
    req_entities: list[str] = []
    if intent_type == "compare":
        # Best-effort concept extraction from the question text.
        # Note: at runtime we don't have the original question string here, only normalized.
        # We accept empty required_entities in that case; compare completeness is enforced by sections.
        req_entities = []
    elif intent_type in {"cross_pillar", "network", "tension", "scenario"}:
        for e in (detected_entities or [])[:8]:
            try:
                conf = float(e.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            if conf < 0.75:
                continue
            name = str(e.get("name_ar") or "").strip()
            if name and (name not in req_entities):
                req_entities.append(name)

    # Required sections vary by intent.
    required_sections: list[str] = []
    if intent_type == "system_limits_policy":
        # Policy questions: no required sections, just needs answer + citations
        required_sections = []
    elif intent_type == "compare":
        required_sections = [
            "مصفوفة المقارنة",
            "خلاصة تنفيذية (3 نقاط)",
        ]
    elif intent_type == "scenario":
        # Stakeholder-friendly: grounded A + explicit unsupported B, no interrogations.
        required_sections = [
            "ما يمكن دعمه من الأدلة المسترجعة",
            "ما لا يمكن الجزم به من الأدلة الحالية",
        ]
    elif intent_type in {"cross_pillar", "network", "tension"}:
        required_sections = [
            "تعريف المفهوم داخل الإطار",
            "التأصيل والأدلة (مختصر ومركز)",
            "الربط بين الركائز (مع سبب الربط)",
            "تطبيق عملي على الحالة/السؤال",
            "تنبيهات وأخطاء شائعة",
            "خلاصة تنفيذية (3 نقاط)",
        ]

    return ContractSpec(
        intent_type=intent_type,
        required_sections=tuple(required_sections),
        required_entities=tuple(req_entities),
        requires_graph=requires_graph,
        min_links=3 if intent_type == "network" else (1 if requires_graph else 0),
        min_distinct_pillars=4 if intent_type == "network" else 0,
        allow_followup_questions=False,
    )


def _iter_section_bullets(answer_ar: str, header: str) -> list[str]:
    t = answer_ar or ""
    if header not in t:
        return []
    lines = [ln.rstrip() for ln in t.splitlines()]
    start = None
    for i, ln in enumerate(lines):
        if header in ln:
            start = i + 1
            break
    if start is None:
        return []
    bullets: list[str] = []
    for ln in lines[start:]:
        s = ln.strip()
        if not s:
            continue
        if (not s.startswith("-")) and (len(s) <= 50) and (" " in s or "داخل" in s or "خلاصة" in s):
            break
        if s.startswith("-"):
            bullets.append(s[1:].strip())
    return bullets


def _has_partial_sections(answer_ar: str) -> bool:
    a_old = "قسم (أ): ما يمكن دعمه من الأدلة المسترجعة"
    b_old = "قسم (ب): ما لا يمكن دعمه من الأدلة الحالية"
    a_new = "ما يمكن دعمه من الأدلة المسترجعة"
    b_new = "ما لا يمكن الجزم به من الأدلة الحالية"
    t = answer_ar or ""
    return ((a_old in t) and (b_old in t)) or ((a_new in t) and (b_new in t))


def _has_explicit_graph_gap_statement(answer_ar: str) -> bool:
    raw = answer_ar or ""
    if ("edge_justification_span" in raw) and ("لا توجد روابط" in raw):
        return True

    t = normalize_for_matching(raw)
    # Conservative: must explicitly mention lack of grounded/justified edges.
    # Note: normalization may strip underscores: "edge_justification_span" -> "edgejustificationspan".
    must_have = ["لا توجد", "روابط"]
    has_root = ("مؤص" in t)  # covers مؤصلة/مؤصله
    has_edge_token = ("edgejustificationspan" in t) or ("edge" in t)
    return all(k in t for k in must_have) and has_root and has_edge_token


def _compare_blocks(answer_ar: str) -> dict[str, dict[str, str]]:
    """
    Parse compare matrix blocks emitted by our deterministic composer.

    Expected format:
    - <concept>:
      - التعريف: ...
      - المظهر العملي: ...
      - الخطأ الشائع: ...
    """

    lines = [ln.rstrip("\n") for ln in (answer_ar or "").splitlines()]
    blocks: dict[str, dict[str, str]] = {}
    cur: Optional[str] = None
    for ln in lines:
        s = ln.strip()
        if s.startswith("- ") and s.endswith(":") and ("مصفوفة" not in s):
            cur = s[2:-1].strip()
            if cur:
                blocks.setdefault(cur, {})
            continue
        if cur and s.startswith("- "):
            kv = s[2:].strip()
            if ":" in kv:
                k, _, v = kv.partition(":")
                k = k.strip()
                v = v.strip()
                if k and v:
                    blocks[cur][k] = v
    return blocks


def _distinct_pillars_in_edges(used_edges: Iterable[UsedEdge]) -> int:
    ps: set[str] = set()
    for e in used_edges:
        for node in [e.from_node, e.to_node]:
            if node.startswith("pillar:"):
                ps.add(node)
    return len(ps)


def check_contract(
    *,
    spec: ContractSpec,
    answer_ar: str,
    citations: list[Any],
    used_edges: list[UsedEdge],
) -> ContractMetrics:
    reasons: list[str] = []

    # Required entities coverage (surface string containment, normalized).
    ans_n = normalize_for_matching(answer_ar or "")
    req = [normalize_for_matching(x) for x in (spec.required_entities or ()) if (x or "").strip()]
    hit = 0
    for r in req:
        if r and (r in ans_n):
            hit += 1
    required_entities_coverage = (hit / len(req)) if req else 1.0
    if required_entities_coverage < 1.0:
        reasons.append("MISSING_REQUIRED_ENTITIES")

    # Required sections must exist and be non-empty (>=1 bullet) when required.
    nonempty = 0
    for h in (spec.required_sections or ()):
        bullets = _iter_section_bullets(answer_ar, h)
        if bullets:
            nonempty += 1
        else:
            reasons.append(f"EMPTY_SECTION:{h}")
    section_nonempty = (nonempty / len(spec.required_sections)) if spec.required_sections else 1.0

    # Compare: concept-complete matrix (definition/practical/confusion per concept).
    if spec.intent_type == "compare" and spec.required_entities:
        blocks = _compare_blocks(answer_ar)
        for concept in spec.required_entities:
            if concept not in blocks:
                reasons.append(f"COMPARE_MISSING_CONCEPT_BLOCK:{concept}")
                continue
            b = blocks[concept]
            for field in ["التعريف", "المظهر العملي", "الخطأ الشائع"]:
                if field not in b:
                    reasons.append(f"COMPARE_MISSING_FIELD:{concept}:{field}")

    # Graph requirements: used_edges reflect exactly relied-upon edges.
    graph_required_satisfied = True
    if spec.requires_graph:
        if (spec.min_links and len(used_edges) < spec.min_links) or (len(used_edges) == 0):
            graph_required_satisfied = False
            reasons.append("MISSING_USED_GRAPH_EDGES")
        if spec.min_distinct_pillars and _distinct_pillars_in_edges(used_edges) < spec.min_distinct_pillars:
            graph_required_satisfied = False
            reasons.append("INSUFFICIENT_DISTINCT_PILLARS_IN_GRAPH")
    else:
        graph_required_satisfied = True

    # Minimal citation presence for non-abstaining answers.
    if (answer_ar or "").strip() and not (citations or []):
        reasons.append("MISSING_CITATIONS")

    # Partial acceptance:
    # If the intent requires a graph but the system cannot produce grounded edges,
    # we accept an explicit A+B partial response (stakeholder-friendly honesty).
    if (not graph_required_satisfied) and spec.intent_type in {"cross_pillar", "network", "tension", "cross_pillar_path"}:
        # Require at least some grounded content in (A).
        if _has_partial_sections(answer_ar) and _has_explicit_graph_gap_statement(answer_ar) and (citations or []):
            outcome = ContractOutcome("PASS_PARTIAL")
            # For stakeholder partials, interpret A+B as satisfying "intent coverage".
            # - Report section_nonempty as 1.0 (partial template present)
            # - Keep graph_required_satisfied=False (no grounded edges were used)
            # - Drop EMPTY_SECTION reasons to avoid confusing metrics (they reflect full template)
            filtered = tuple([r for r in reasons if not r.startswith("EMPTY_SECTION:")])
            return ContractMetrics(
                outcome=outcome,
                reasons=filtered,
                section_nonempty=1.0,
                required_entities_coverage=required_entities_coverage,
                graph_required_satisfied=False,
            )

    ok = (len(reasons) == 0)
    outcome = ContractOutcome("PASS_FULL" if ok else "FAIL")
    return ContractMetrics(
        outcome=outcome,
        reasons=tuple(reasons),
        section_nonempty=section_nonempty,
        required_entities_coverage=required_entities_coverage,
        graph_required_satisfied=graph_required_satisfied,
    )

