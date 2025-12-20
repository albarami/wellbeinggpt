"""Runtime answer-contract gate (shared logic with eval).

This enforces stakeholder intent *at runtime*, not only in eval.

Behavior:
- Build ContractSpec deterministically from the runtime question + detected entities.
- Check contract against the produced answer + citations + used semantic edges.
- If contract fails:
  - Run one repair attempt (targeted retrieval + re-compose) when possible
  - Otherwise: partial scenario output (A+B) or fail-closed abstention

Reason: stakeholder acceptance showed "safe but off-target" outputs in production flow.
"""

from __future__ import annotations

from typing import Any

from apps.api.core.answer_contract import UsedEdge, UsedEdgeSpan, check_contract, contract_from_question_runtime
from apps.api.core.schemas import Confidence


def _used_edges_from_middleware(middleware) -> list[UsedEdge]:
    out: list[UsedEdge] = []
    used = list(getattr(middleware, "_last_used_edges", None) or [])
    for ue in used:
        spans: list[UsedEdgeSpan] = []
        for sp in list(ue.get("justification_spans") or [])[:8]:
            spans.append(
                UsedEdgeSpan(
                    chunk_id=str(sp.get("chunk_id") or ""),
                    span_start=int(sp.get("span_start") or 0),
                    span_end=int(sp.get("span_end") or 0),
                    quote=str(sp.get("quote") or ""),
                )
            )
        out.append(
            UsedEdge(
                edge_id=str(ue.get("edge_id") or ""),
                from_node=str(ue.get("from_node") or ""),
                to_node=str(ue.get("to_node") or ""),
                relation_type=str(ue.get("relation_type") or ""),
                justification_spans=tuple(spans),
            )
        )
    return out


async def apply_runtime_contract_gate(
    *,
    middleware,
    ctx,
    enable_repair: bool = True,
) -> None:
    """
    Apply contract gate to the runtime Muḥāsibī context.

    This function mutates `ctx` in-place.
    """

    # If we already refused, do not override.
    if bool(getattr(ctx, "not_found", False)):
        return

    # Contract scope gate:
    # Only enforce runtime contracts when explicit answer_requirements are present.
    # Reason: otherwise we "fail closed" on ordinary questions with no declared structure,
    # recreating the stakeholder over-abstention failure mode.
    ar = getattr(ctx, "answer_requirements", None)
    ar_dict = dict(ar) if isinstance(ar, dict) else {}
    applicable = any(k in ar_dict for k in ["must_include", "path_trace", "format"])
    if not applicable:
        # Observability (UI support): mark gate as not applied.
        try:
            setattr(ctx, "contract_gate_debug", {"applied": False})
        except Exception:
            pass
        return

    q_norm = str(getattr(ctx, "normalized_question", "") or "")
    detected = list(getattr(ctx, "detected_entities", None) or [])
    spec = contract_from_question_runtime(question_norm=q_norm, detected_entities=detected)

    used_edges = _used_edges_from_middleware(middleware)
    answer_ar = str(getattr(ctx, "answer_ar", "") or "")
    citations = list(getattr(ctx, "citations", None) or [])

    cm = check_contract(spec=spec, answer_ar=answer_ar, citations=citations, used_edges=used_edges)
    try:
        setattr(
            ctx,
            "contract_gate_debug",
            {
                "applied": True,
                "intent_type": getattr(spec, "intent_type", "generic"),
                "outcome": cm.outcome.value,
                "reasons": list(cm.reasons or ()),
                "repair_attempted": False,
                "repair_succeeded": False,
                "partial_emitted": False,
                "fail_closed": False,
            },
        )
    except Exception:
        pass
    if cm.outcome.value in {"PASS_FULL", "PASS_PARTIAL"}:
        return

    # Scenario: always prefer partial grounded A+B (no interrogations).
    if spec.intent_type == "scenario":
        try:
            from apps.api.core.scholar_reasoning_compose_scenario import compose_partial_scenario_answer

            packets = list(getattr(ctx, "evidence_packets", None) or [])
            # Natural chat: route partial through the natural writer when available (keeps voice consistent).
            if (getattr(ctx, "mode", "") or "") == "natural_chat" and getattr(middleware, "llm_client", None) is not None:
                try:
                    llm = getattr(middleware, "llm_client", None)
                    res = await llm.interpret(
                        question=str(getattr(ctx, "question", "") or ""),
                        evidence_packets=packets[:18],
                        detected_entities=list(getattr(ctx, "detected_entities", None) or [])[:8],
                        mode="natural_chat",
                        used_edges=[],
                        argument_chains=[],
                        fallback_context={
                            "partial_required": True,
                            "required_markers_ar": [
                                "ما يمكن دعمه من الأدلة المسترجعة",
                                "ما لا يمكن الجزم به من الأدلة الحالية",
                            ],
                        },
                    )
                    if res and str(res.answer_ar or "").strip() and list(res.citations or []):
                        from apps.api.core.schemas import Citation

                        ctx.answer_ar = str(res.answer_ar or "")
                        ctx.citations = [
                            Citation(chunk_id=c.get("chunk_id", ""), source_anchor=c.get("source_anchor") or "", ref=c.get("ref"))
                            for c in (res.citations or [])
                            if c.get("chunk_id")
                        ]
                        ctx.not_found = bool(res.not_found)
                        ctx.confidence = Confidence.MEDIUM if not ctx.not_found else Confidence.LOW
                        try:
                            dbg = dict(getattr(ctx, "contract_gate_debug", {}) or {})
                            dbg["partial_emitted"] = True
                            setattr(ctx, "contract_gate_debug", dbg)
                        except Exception:
                            pass
                        return
                except Exception:
                    pass

            ans, cite_objs, _ = compose_partial_scenario_answer(packets=packets, question_ar=str(getattr(ctx, "question", "") or ""), prefer_more_claims=False)
            if (ans or "").strip() and cite_objs:
                ctx.answer_ar = ans
                ctx.citations = cite_objs
                ctx.not_found = False
                ctx.confidence = Confidence.MEDIUM
                try:
                    dbg = dict(getattr(ctx, "contract_gate_debug", {}) or {})
                    dbg["partial_emitted"] = True
                    setattr(ctx, "contract_gate_debug", dbg)
                except Exception:
                    pass
                return
        except Exception:
            pass

    # Repair loop (one pass) before abstaining.
    if enable_repair:
        try:
            dbg = dict(getattr(ctx, "contract_gate_debug", {}) or {})
            dbg["repair_attempted"] = True
            setattr(ctx, "contract_gate_debug", dbg)
        except Exception:
            pass
        try:
            if bool(getattr(ctx, "deep_mode", False)) and getattr(middleware, "retriever", None) is not None:
                session = getattr(getattr(middleware, "retriever", None), "_session", None)
                if session:
                    from apps.api.core.scholar_reasoning import ScholarReasoner

                    reasoner = ScholarReasoner(session=session, retriever=middleware.retriever)
                    res = await reasoner.generate(
                        question=str(getattr(ctx, "question", "") or ""),
                        question_norm=str(getattr(ctx, "normalized_question", "") or ""),
                        detected_entities=detected,
                        evidence_packets=list(getattr(ctx, "evidence_packets", None) or []),
                    )
                    if res and str(res.get("answer_ar") or "").strip() and list(res.get("citations") or []):
                        ctx.answer_ar = str(res.get("answer_ar") or "")
                        ctx.citations = list(res.get("citations") or [])
                        try:
                            setattr(middleware, "_last_used_edges", list(res.get("used_edges") or []))
                        except Exception:
                            setattr(middleware, "_last_used_edges", [])
                        ctx.not_found = bool(res.get("not_found"))
                        ctx.confidence = res.get("confidence") or Confidence.MEDIUM

                        used_edges2 = _used_edges_from_middleware(middleware)
                        cm2 = check_contract(
                            spec=spec,
                            answer_ar=str(getattr(ctx, "answer_ar", "") or ""),
                            citations=list(getattr(ctx, "citations", None) or []),
                            used_edges=used_edges2,
                        )
                        if cm2.outcome.value in {"PASS_FULL", "PASS_PARTIAL"} and not bool(getattr(ctx, "not_found", False)):
                            try:
                                dbg = dict(getattr(ctx, "contract_gate_debug", {}) or {})
                                dbg["repair_succeeded"] = True
                                setattr(ctx, "contract_gate_debug", dbg)
                            except Exception:
                                pass
                            return
        except Exception:
            pass

    # If we failed specifically due to missing grounded graph edges, prefer a partial A+B
    # for graph-centric intents (stakeholder-friendly honesty).
    if spec.intent_type in {"cross_pillar", "network", "tension", "cross_pillar_path"} and "MISSING_USED_GRAPH_EDGES" in cm.reasons:
        try:
            from apps.api.core.scholar_reasoning_compose_partial_graph import compose_partial_graph_gap_answer

            packets = list(getattr(ctx, "evidence_packets", None) or [])
            ans, cite_objs, _ = compose_partial_graph_gap_answer(packets=packets, question_ar=str(getattr(ctx, "question", "") or ""))
            if (ans or "").strip() and cite_objs:
                ctx.answer_ar = ans
                ctx.citations = cite_objs
                ctx.not_found = False
                ctx.confidence = Confidence.MEDIUM
                try:
                    dbg = dict(getattr(ctx, "contract_gate_debug", {}) or {})
                    dbg["partial_emitted"] = True
                    setattr(ctx, "contract_gate_debug", dbg)
                except Exception:
                    pass
                return
        except Exception:
            pass

    # Fail-closed: contract unmet (only after partial/repair).
    ctx.answer_ar = "لا يوجد في البيانات الحالية ما يدعم إجابة ملتزمة بعقد السؤال."
    ctx.citations = []
    ctx.not_found = True
    ctx.confidence = Confidence.LOW
    try:
        dbg = dict(getattr(ctx, "contract_gate_debug", {}) or {})
        dbg["fail_closed"] = True
        setattr(ctx, "contract_gate_debug", dbg)
    except Exception:
        pass

