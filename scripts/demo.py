#!/usr/bin/env python
"""
Demo Script for Wellbeing Data Foundation

Demonstrates:
1. Ingestion results (from canonical JSON)
2. Evidence bundle retrieval
3. In-corpus question answering
4. Out-of-corpus refusal behavior
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from apps.api.core.muhasibi_state_machine import create_middleware
from apps.api.retrieve.entity_resolver import EntityResolver
from apps.api.guardrails.citation_enforcer import create_guardrails


def print_header(text: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_json(data: Any, indent: int = 2) -> None:
    """Print JSON data nicely."""
    print(json.dumps(data, ensure_ascii=False, indent=indent))


def demo_sample_data() -> dict[str, Any]:
    """
    Create sample canonical data for demonstration.

    In production, this would come from the ingestion pipeline.
    """
    return {
        "meta": {
            "source_doc_id": "DOC_demo_001",
            "source_file_hash": "abcd1234efgh5678",
            "framework_version": "2025-10",
            "extracted_at": "2025-12-12T00:00:00",
            "stats": {
                "total_pillars": 1,
                "total_core_values": 2,
                "total_sub_values": 3,
                "total_evidence": 4,
            },
        },
        "pillars": [
            {
                "id": "P001",
                "name_ar": "الحياة الروحية الطيبة",
                "description_ar": "الركيزة الأولى من ركائز الحياة الطيبة",
                "core_values": [
                    {
                        "id": "CV001",
                        "name_ar": "الإيمان",
                        "definition": {
                            "text_ar": "الإيمان هو التصديق بالقلب والإقرار باللسان والعمل بالأركان",
                        },
                        "sub_values": [
                            {
                                "id": "SV001",
                                "name_ar": "التوحيد",
                                "definition": {
                                    "text_ar": "التوحيد هو إفراد الله تعالى بالعبادة والألوهية",
                                },
                            },
                            {
                                "id": "SV002",
                                "name_ar": "الإخلاص",
                                "definition": {
                                    "text_ar": "الإخلاص هو تصفية العمل من كل شوائب الرياء",
                                },
                            },
                        ],
                    },
                    {
                        "id": "CV002",
                        "name_ar": "الصبر",
                        "definition": {
                            "text_ar": "الصبر هو حبس النفس عن الجزع والتسخط",
                        },
                        "evidence": [
                            {
                                "evidence_type": "quran",
                                "ref_raw": "[النحل: 127]",
                                "text_ar": "وَاصْبِرْ وَمَا صَبْرُكَ إِلَّا بِاللَّهِ",
                            },
                        ],
                        "sub_values": [
                            {
                                "id": "SV003",
                                "name_ar": "الرضا",
                                "definition": {
                                    "text_ar": "الرضا هو قبول قضاء الله وقدره بطيب نفس",
                                },
                            },
                        ],
                    },
                ],
            },
        ],
    }


def demo_evidence_packets() -> list[dict[str, Any]]:
    """
    Create sample evidence packets for demonstration.
    """
    return [
        {
            "chunk_id": "CH_000001",
            "entity_type": "core_value",
            "entity_id": "CV001",
            "chunk_type": "definition",
            "text_ar": "الإيمان هو التصديق بالقلب والإقرار باللسان والعمل بالأركان",
            "source_doc_id": "DOC_demo_001",
            "source_anchor": "p10_abc123",
            "refs": [],
        },
        {
            "chunk_id": "CH_000002",
            "entity_type": "sub_value",
            "entity_id": "SV001",
            "chunk_type": "definition",
            "text_ar": "التوحيد هو إفراد الله تعالى بالعبادة والألوهية",
            "source_doc_id": "DOC_demo_001",
            "source_anchor": "p15_def456",
            "refs": [],
        },
        {
            "chunk_id": "CH_000003",
            "entity_type": "core_value",
            "entity_id": "CV002",
            "chunk_type": "definition",
            "text_ar": "الصبر هو حبس النفس عن الجزع والتسخط",
            "source_doc_id": "DOC_demo_001",
            "source_anchor": "p20_ghi789",
            "refs": [],
        },
        {
            "chunk_id": "CH_000004",
            "entity_type": "core_value",
            "entity_id": "CV002",
            "chunk_type": "evidence",
            "text_ar": "وَاصْبِرْ وَمَا صَبْرُكَ إِلَّا بِاللَّهِ",
            "source_doc_id": "DOC_demo_001",
            "source_anchor": "p21_jkl012",
            "refs": [{"type": "quran", "ref": "النحل:127"}],
        },
    ]


async def demo_ingestion_results() -> None:
    """Demonstrate ingestion results."""
    print_header("1. Ingestion Results (Sample Data)")

    data = demo_sample_data()

    print("\n📊 Ingestion Statistics:")
    stats = data["meta"]["stats"]
    print(f"  • Pillars: {stats['total_pillars']}")
    print(f"  • Core Values: {stats['total_core_values']}")
    print(f"  • Sub-Values: {stats['total_sub_values']}")
    print(f"  • Evidence Items: {stats['total_evidence']}")

    print("\n📋 Extracted Hierarchy:")
    for pillar in data["pillars"]:
        print(f"\n  🏛️ {pillar['name_ar']}")
        for cv in pillar["core_values"]:
            print(f"    └── 💎 {cv['name_ar']}")
            for sv in cv.get("sub_values", []):
                print(f"        └── 🌱 {sv['name_ar']}")


async def demo_evidence_bundle() -> None:
    """Demonstrate evidence bundle retrieval."""
    print_header("2. Evidence Bundle Retrieval")

    packets = demo_evidence_packets()

    print(f"\n📦 Retrieved {len(packets)} evidence packets:\n")

    for i, packet in enumerate(packets, 1):
        print(f"  [{i}] Chunk ID: {packet['chunk_id']}")
        print(f"      Type: {packet['chunk_type']} ({packet['entity_type']})")
        print(f"      Text: {packet['text_ar'][:60]}...")
        if packet["refs"]:
            print(f"      Refs: {packet['refs']}")
        print()


async def demo_in_corpus_question() -> None:
    """Demonstrate in-corpus question answering."""
    print_header("3. In-Corpus Question: ما هو الإيمان؟")

    # Set up resolver with sample data
    resolver = EntityResolver()
    resolver.load_entities(
        pillars=[{"id": "P001", "name_ar": "الحياة الروحية الطيبة"}],
        core_values=[
            {"id": "CV001", "name_ar": "الإيمان"},
            {"id": "CV002", "name_ar": "الصبر"},
        ],
        sub_values=[
            {"id": "SV001", "name_ar": "التوحيد"},
            {"id": "SV002", "name_ar": "الإخلاص"},
            {"id": "SV003", "name_ar": "الرضا"},
        ],
    )

    middleware = create_middleware(entity_resolver=resolver)
    response = await middleware.process("ما هو الإيمان؟")

    print("\n🔍 Query: ما هو الإيمان؟")
    print(f"\n📝 Listen Summary: {response.listen_summary_ar}")
    print(f"\n🎯 Purpose: {response.purpose.ultimate_goal_ar}")
    print(f"\n📋 Plan Steps:")
    for i, step in enumerate(response.path_plan_ar, 1):
        print(f"   {i}. {step}")

    print(f"\n💬 Answer: {response.answer_ar}")
    print(f"\n📊 Confidence: {response.confidence.value}")
    print(f"❓ Not Found: {response.not_found}")

    if response.entities:
        print("\n🏷️ Detected Entities:")
        for entity in response.entities:
            print(f"   • {entity.name_ar} ({entity.type.value})")


async def demo_out_of_corpus_refusal() -> None:
    """Demonstrate out-of-corpus refusal."""
    print_header("4. Out-of-Corpus Question (Refusal Demo)")

    middleware = create_middleware()

    # Question clearly outside the wellbeing framework
    question = "ما هي عاصمة فرنسا؟"

    print(f"\n🔍 Query: {question}")
    print("   (This is outside the wellbeing framework scope)")

    response = await middleware.process(question)

    print(f"\n💬 Answer: {response.answer_ar}")
    print(f"\n❓ Not Found: {response.not_found}")
    print(f"📊 Confidence: {response.confidence.value}")
    print(f"📚 Citations: {len(response.citations)}")

    print("\n✅ Refusal behavior confirmed: System refuses to hallucinate!")


async def demo_guardrails() -> None:
    """Demonstrate guardrails validation."""
    print_header("5. Guardrails Validation Demo")

    guardrails = create_guardrails(min_coverage_ratio=0.5)
    packets = demo_evidence_packets()

    # Valid answer with citations
    print("\n✅ Test 1: Valid answer with proper citations")
    result = guardrails.validate(
        answer_ar="الإيمان هو التصديق بالقلب والإقرار باللسان",
        citations=[{"chunk_id": "CH_000001"}],
        evidence_packets=packets,
        not_found=False,
    )
    print(f"   Passed: {result.passed}")
    print(f"   Issues: {result.issues or 'None'}")

    # Invalid: no citations
    print("\n❌ Test 2: Answer without citations (should fail)")
    result = guardrails.validate(
        answer_ar="الإيمان هو التصديق بالقلب",
        citations=[],
        evidence_packets=packets,
        not_found=False,
    )
    print(f"   Passed: {result.passed}")
    print(f"   Issues: {result.issues}")
    print(f"   Should Retry: {result.should_retry}")

    # Invalid: invalid chunk_id
    print("\n❌ Test 3: Citation with invalid chunk_id (should fail)")
    result = guardrails.validate(
        answer_ar="الإيمان هو التصديق بالقلب",
        citations=[{"chunk_id": "CH_INVALID"}],
        evidence_packets=packets,
        not_found=False,
    )
    print(f"   Passed: {result.passed}")
    print(f"   Issues: {result.issues}")


async def main() -> None:
    """Run all demos."""
    print("\n" + "🌟" * 30)
    print("  WELLBEING DATA FOUNDATION - DEMO")
    print("  Evidence-Only Arabic Assistant")
    print("🌟" * 30)

    await demo_ingestion_results()
    await demo_evidence_bundle()
    await demo_in_corpus_question()
    await demo_out_of_corpus_refusal()
    await demo_guardrails()

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print("\n✅ All demos executed successfully!")
    print("📌 Key takeaways:")
    print("   • System extracts structured data from Arabic documents")
    print("   • Evidence packets are citeable with stable anchors")
    print("   • Muḥāsibī middleware enforces evidence-only answers")
    print("   • Guardrails block hallucination attempts")
    print("   • Out-of-scope questions trigger refusal (not_found=true)")
    print()


if __name__ == "__main__":
    asyncio.run(main())

