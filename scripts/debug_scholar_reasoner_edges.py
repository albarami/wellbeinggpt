"""Debug: run ScholarReasoner on a cross-pillar question and print used_edges."""

from __future__ import annotations

import asyncio
import sys

from apps.api.core.database import get_session
from apps.api.core.scholar_reasoning import ScholarReasoner
from apps.api.retrieve.hybrid_retriever import HybridRetriever
from apps.api.retrieve.normalize_ar import normalize_for_matching
from eval.datasets.source_loader import load_dotenv_if_present
from apps.api.core.scholar_reasoning_edge_fallback import semantic_edges_fallback


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
        retriever = HybridRetriever()
        reasoner = ScholarReasoner(session=session, retriever=retriever)
        se = await semantic_edges_fallback(session=session, question_norm=normalize_for_matching(q), max_edges=8, extra=False)
        print("fallback_semantic_edges:", len(se))
        if se:
            print("first_edge:", {k: se[0].get(k) for k in ["edge_id", "relation_type", "neighbor_type", "neighbor_id", "direction", "justification_spans", "source_id"]})
        res = await reasoner.generate(
            question=q,
            question_norm=normalize_for_matching(q),
            detected_entities=[],
            evidence_packets=[],
        )
        print("not_found:", res.get("not_found"), "confidence:", res.get("confidence"))
        ue = list(res.get("used_edges") or [])
        print("used_edges:", len(ue))
        for e in ue:
            print(e)
        print("\nanswer_ar:\n", str(res.get("answer_ar") or "")[:700])


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

