"""Debug: run framework edge miner on a specific chunk_id and print results."""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.graph.framework_edge_miner import extract_semantic_edges_from_chunk
from apps.api.retrieve.normalize_ar import normalize_for_matching
from apps.api.ingest.sentence_spans import sentence_spans, span_text
from eval.datasets.source_loader import load_dotenv_if_present


async def _run() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    load_dotenv_if_present()
    target = "CH_00df4b4c8b27"
    async with get_session() as session:
        row = (
            await session.execute(
                text("SELECT chunk_id, chunk_type, text_ar FROM chunk WHERE chunk_id=:cid LIMIT 1"),
                {"cid": target},
            )
        ).fetchone()
        if not row:
            print("chunk_not_found")
            return
        txt = str(row.text_ar or "")
        tnorm = normalize_for_matching(txt)
        # Quick probes for key substrings (normalized + raw).
        probes = [
            "وسيلة",
            "إطلاق",
            "اطلاق",
            "الروحية",
            "العاطفية",
            "العقلية",
            "الاجتماعية",
            "لا يفصل",
            "الجسد",
            "الروح",
        ]
        print("probe_hits_raw:", {p: (p in txt) for p in probes})
        print("probe_hits_norm:", {p: (p in tnorm) for p in probes})
        # Show first 3 sentence spans + detected pillar ids for each.
        for i, sp in enumerate(sentence_spans(txt)[:3], start=1):
            sent = span_text(txt, sp).strip()
            sn = normalize_for_matching(sent)
            from apps.api.graph.framework_edge_miner import _pillars_mentioned  # type: ignore

            print(f"SPAN{i} [{sp.start}:{sp.end}] len={len(sent)}")
            print(sent[:220].replace("\n", " "))
            print("pillars:", _pillars_mentioned(sent))
            print("has_marker_وسيلة:", "وسيلة" in sn, "has_marker_اطلاق:", ("اطلاق" in sn))
        mined = extract_semantic_edges_from_chunk(chunk_id=str(row.chunk_id), text_ar=txt)
        print(f"chunk_id={row.chunk_id} chunk_type={row.chunk_type} mined_edges={len(mined)}")
        for m in mined[:20]:
            print(f"- {m.from_pillar_id}->{m.to_pillar_id} {m.relation_type} [{m.span_start}:{m.span_end}] {m.quote[:120]}")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

