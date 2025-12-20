"""Debug: simulate eval claim binding + pruning for stakeholder #2 answer."""

from __future__ import annotations

import asyncio
import sys

from apps.api.core.database import get_session
from apps.api.core.scholar_reasoning import ScholarReasoner
from apps.api.retrieve.hybrid_retriever import HybridRetriever
from apps.api.retrieve.normalize_ar import normalize_for_matching
from eval.citations import citation_for_chunk_best_sentence
from eval.claims import extract_claims
from eval.prune import prune_and_fail_closed
from eval.types import EvalCitation, EvalMode
from eval.datasets.source_loader import load_dotenv_if_present


async def _run() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    load_dotenv_if_present()
    q = (
        "اشرح كيف يمكن لقيمة من الركيزة البدنية أن تكون مُعينة/مُمكِّنة لقيمة من الركيزة الروحية في هذا الإطار. "
        "قدّم مسار الربط خطوة بخطوة مع تبرير كل خطوة (سبب الربط) وبأدلة لكل رابط."
    )
    async with get_session() as session:
        reasoner = ScholarReasoner(session=session, retriever=HybridRetriever())
        res = await reasoner.generate(
            question=q,
            question_norm=normalize_for_matching(q),
            detected_entities=[],
            evidence_packets=[],
        )
        ans = str(res.get("answer_ar") or "")
        used_edges = list(res.get("used_edges") or [])
        cite_ids = [getattr(c, "chunk_id", "") for c in (res.get("citations") or [])]

        citations: list[EvalCitation] = []
        for cid in cite_ids:
            c = await citation_for_chunk_best_sentence(session, chunk_id=cid, query_text=q)
            if c is not None:
                citations.append(c)

        # Add exact edge justification spans as citations too.
        for ue in used_edges:
            for sp in list(ue.get("justification_spans") or [])[:6]:
                cid = str(sp.get("chunk_id") or "")
                citations.append(
                    EvalCitation(
                        source_id=cid,
                        span_start=int(sp.get("span_start") or 0),
                        span_end=int(sp.get("span_end") or 0),
                        quote=str(sp.get("quote") or ""),
                    )
                )

        claims = extract_claims(answer_ar=ans, mode=EvalMode.FULL_SYSTEM, citations=citations)
        with_spans = sum(1 for c in claims if (c.evidence.supporting_spans or []))
        print("answer_len:", len(ans), "claims:", len(claims), "claims_with_spans:", with_spans, "citations:", len(citations))
        pr = await prune_and_fail_closed(
            session=session,
            mode=EvalMode.FULL_SYSTEM,
            answer_ar=ans,
            claims=claims,
            citations=citations,
            question_ar=q,
            resolved_entities=[],
            required_graph_paths=None,
        )
        print("pruned_abstained:", pr.abstained, "reason:", pr.abstain_reason, "kept_claims:", len(pr.claims))
        print("\nPRUNED_ANSWER_PREVIEW:\n", pr.answer_ar[:500])


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

