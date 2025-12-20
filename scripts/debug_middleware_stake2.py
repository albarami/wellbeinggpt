"""Debug: run Muḥāsibī middleware for stake-0002 and inspect citations."""

from __future__ import annotations

import asyncio
import sys

from apps.api.core.database import get_session
from apps.api.core.muhasibi_state_machine import create_middleware
from apps.api.guardrails.citation_enforcer import Guardrails
from apps.api.retrieve.hybrid_retriever import HybridRetriever
from eval.datasets.source_loader import load_dotenv_if_present
from eval.runner_helpers import build_entity_resolver


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
        resolver = await build_entity_resolver(session)
        retriever = HybridRetriever(enable_graph=True)
        retriever._session = session  # type: ignore[attr-defined]
        mw = create_middleware(entity_resolver=resolver, retriever=retriever, llm_client=None, guardrails=Guardrails())
        final = await mw.process(q, language="ar", mode="answer")
        print("not_found:", final.not_found, "confidence:", final.confidence)
        cits = list(final.citations or [])
        print("citations:", len(cits))
        if cits:
            print("first_citation:", {"chunk_id": cits[0].chunk_id, "source_anchor": cits[0].source_anchor})
        print("answer_preview:\n", (final.answer_ar or "")[:800])


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

