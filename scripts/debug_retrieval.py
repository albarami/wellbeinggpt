#!/usr/bin/env python3
"""Debug retrieval for pillar question."""

import sys
import os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from sqlalchemy import text
from apps.api.retrieve.entity_resolver import EntityResolver
from apps.api.retrieve.hybrid_retriever import HybridRetriever, RetrievalInputs
from apps.api.core.database import get_session


async def main():
    question = "اذكر ركائز الحياة الطيبة الخمس."
    
    async with get_session() as session:
        # 1. Load entity resolver
        resolver = EntityResolver()
        pillars = (await session.execute(text("SELECT id, name_ar FROM pillar"))).fetchall()
        core_values = (await session.execute(text("SELECT id, name_ar FROM core_value"))).fetchall()
        sub_values = (await session.execute(text("SELECT id, name_ar FROM sub_value"))).fetchall()
        
        print(f"=== Database Contents ===")
        print(f"Pillars in DB: {len(pillars)}")
        for p in pillars:
            print(f"  - {p.name_ar} (id={p.id})")
        print(f"Core values in DB: {len(core_values)}")
        print(f"Sub values in DB: {len(sub_values)}")
        
        resolver.load_entities(
            pillars=[{"id": r.id, "name_ar": r.name_ar} for r in pillars],
            core_values=[{"id": r.id, "name_ar": r.name_ar} for r in core_values],
            sub_values=[{"id": r.id, "name_ar": r.name_ar} for r in sub_values],
            aliases_path="data/static/aliases_ar.json",
        )
        
        print(f"\n=== Entity Resolution ===")
        print(f"Question: {question}")
        resolved = resolver.resolve(question)
        print(f"Resolved entities: {len(resolved)}")
        for r in resolved[:10]:
            print(f"  - {r.name_ar} ({r.entity_type.value}) confidence={r.confidence:.2f}")
        
        # 2. Check retrieval
        print(f"\n=== Retrieval ===")
        retriever = HybridRetriever()
        retriever._session = session
        
        entity_dicts = [
            {
                "type": r.entity_type.value,
                "id": r.entity_id,
                "name_ar": r.name_ar,
                "confidence": r.confidence,
            }
            for r in resolved
        ]
        
        merge = await retriever.retrieve(
            session,
            RetrievalInputs(
                query=question,
                resolved_entities=entity_dicts,
            ),
        )
        
        print(f"Evidence packets: {len(merge.evidence_packets)}")
        print(f"Has definition: {merge.has_definition}")
        print(f"Has evidence: {merge.has_evidence}")
        
        for i, p in enumerate(merge.evidence_packets[:10]):
            print(f"\n--- Packet {i+1} ---")
            print(f"  chunk_id: {p.get('chunk_id')}")
            print(f"  entity_id: {p.get('entity_id')}")
            print(f"  entity_name_ar: {p.get('entity_name_ar', 'N/A')}")
            print(f"  entity_type: {p.get('entity_type')}")
            print(f"  chunk_type: {p.get('chunk_type')}")
            print(f"  text_ar (first 200): {p.get('text_ar', '')[:200]}")


if __name__ == "__main__":
    asyncio.run(main())




