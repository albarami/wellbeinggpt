"""
Vector Coverage Check Script

Verifies that entity definition chunks are retrievable via the configured retrieval backend.

In this environment we support `VECTOR_BACKEND=bm25` (local, no Azure Search).

Usage:
    python scripts/vector_coverage_check.py [--top-k 10]

Env vars required:
    DATABASE_URL - Postgres connection string
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from dotenv import load_dotenv

from apps.api.core.database import get_session
from apps.api.retrieve.vector_retriever import VectorRetriever


def _require_env(name: str) -> str:
    val = (os.getenv(name) or "").strip()
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


async def run_vector_coverage_check(top_k: int = 10) -> dict[str, Any]:
    """
    Run vector coverage check for all entities.
    
    Args:
        top_k: Number of results to check for each query.
        
    Returns:
        Report dictionary.
    """
    report: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "top_k": top_k,
        "vector_backend": os.getenv("VECTOR_BACKEND", "disabled").lower(),
        "entities_checked": 0,
        "entities_found_in_top_k": 0,
        "failures": [],
        "ref_query_checks": [],
        "status": "pending",
    }
    
    retriever = VectorRetriever()
    
    async with get_session() as session:
        # Get all entities with their definition chunks
        entity_queries = [
            ("pillar", """
                SELECT p.id, p.name_ar, c.chunk_id
                FROM pillar p
                JOIN chunk c ON c.entity_type = 'pillar' 
                    AND c.entity_id = p.id 
                    AND c.chunk_type = 'definition'
            """),
            ("core_value", """
                SELECT cv.id, cv.name_ar, c.chunk_id
                FROM core_value cv
                JOIN chunk c ON c.entity_type = 'core_value' 
                    AND c.entity_id = cv.id 
                    AND c.chunk_type = 'definition'
            """),
            ("sub_value", """
                SELECT sv.id, sv.name_ar, c.chunk_id
                FROM sub_value sv
                JOIN chunk c ON c.entity_type = 'sub_value' 
                    AND c.entity_id = sv.id 
                    AND c.chunk_type = 'definition'
            """),
        ]
        
        for entity_type, query in entity_queries:
            rows = (await session.execute(text(query))).fetchall()
            
            for row in rows:
                entity_id = str(row.id)
                name_ar = str(row.name_ar)
                expected_chunk_id = str(row.chunk_id)
                
                report["entities_checked"] += 1
                
                try:
                    # Search using entity name as query
                    results = await retriever.search(
                        session,
                        query=name_ar,
                        top_k=top_k,
                    )
                    
                    # Check if expected chunk is in results
                    found_chunk_ids = [r.get("chunk_id") for r in results]
                    
                    # Accept ANY definition chunk for this entity in top-K (heading chunk or full definition).
                    expected_ids = (
                        await session.execute(
                            text(
                                """
                                SELECT chunk_id
                                FROM chunk
                                WHERE entity_type = :et AND entity_id = :eid AND chunk_type = 'definition'
                                """
                            ),
                            {"et": entity_type, "eid": entity_id},
                        )
                    ).fetchall()
                    expected_any = {str(r.chunk_id) for r in expected_ids}

                    if any(cid in expected_any for cid in found_chunk_ids):
                        report["entities_found_in_top_k"] += 1
                    else:
                        report["failures"].append({
                            "entity_type": entity_type,
                            "entity_id": entity_id,
                            "name_ar": name_ar,
                            "expected_chunk_id": expected_chunk_id,
                            "expected_chunk_ids_any": sorted(list(expected_any))[:10],
                            "returned_chunk_ids": found_chunk_ids[:5],
                            "reason": f"Expected chunk not in top-{top_k}",
                        })
                except Exception as e:
                    report["failures"].append({
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "name_ar": name_ar,
                        "expected_chunk_id": expected_chunk_id,
                        "reason": f"Search error: {str(e)}",
                    })

        # Additional reference-query spot checks (proves evidence chunks are retrievable by ref strings).
        # These do NOT hardcode answers; they validate retrieval returns the expected chunk IDs.
        ref_spot_checks = [
            {"id": "ref-01", "query": "آل عمران: 200"},
            {"id": "ref-02", "query": "يوسف: 86"},
            {"id": "ref-03", "query": "الحجرات: 6"},
        ]

        for chk in ref_spot_checks:
            q = chk["query"]
            # Identify at least one expected chunk_id in DB that has a matching chunk_ref.ref containing the query.
            expected = (
                await session.execute(
                    text(
                        """
                        SELECT DISTINCT cr.chunk_id
                        FROM chunk_ref cr
                        WHERE cr.ref ILIKE '%' || :q || '%'
                        LIMIT 10
                        """
                    ),
                    {"q": q},
                )
            ).fetchall()
            expected_ids = [str(r.chunk_id) for r in expected]
            if not expected_ids:
                report["ref_query_checks"].append(
                    {
                        "id": chk["id"],
                        "query": q,
                        "status": "no_expected_chunks_in_db",
                        "expected_chunk_ids": [],
                        "returned_chunk_ids": [],
                    }
                )
                continue

            results = await retriever.search(session, query=q, top_k=max(3, top_k))
            returned_ids = [str(r.get("chunk_id")) for r in results if r.get("chunk_id")]
            ok = any(eid in returned_ids[: max(3, top_k)] for eid in expected_ids)
            report["ref_query_checks"].append(
                {
                    "id": chk["id"],
                    "query": q,
                    "status": "pass" if ok else "fail",
                    "expected_chunk_ids": expected_ids[:10],
                    "returned_chunk_ids": returned_ids[:10],
                }
            )
    
    # Determine status
    if report["entities_checked"] == 0:
        report["status"] = "error"
        report["error"] = "No entities with definition chunks found"
    elif len(report["failures"]) == 0 and all(c.get("status") in ("pass", "no_expected_chunks_in_db") for c in report["ref_query_checks"]):
        report["status"] = "complete"
    else:
        report["status"] = "incomplete"
    
    return report


def main() -> None:
    """Main entry point."""
    load_dotenv()
    _require_env("DATABASE_URL")
    parser = argparse.ArgumentParser(description="Vector coverage check")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K to check (default: 10)")
    args = parser.parse_args()
    
    print(f"Running vector coverage check (top-{args.top_k})...")
    print("=" * 60)
    
    report = asyncio.run(run_vector_coverage_check(top_k=args.top_k))
    
    # Write report
    output_path = Path("vector_coverage_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"Report written to: {output_path}")
    print()
    print("SUMMARY:")
    print(f"  Status: {report['status']}")
    print(f"  Entities checked: {report['entities_checked']}")
    print(f"  Found in top-{args.top_k}: {report['entities_found_in_top_k']}")
    print(f"  Failures: {len(report['failures'])}")
    print()
    
    if report.get("error"):
        print(f"ERROR: {report['error']}")
        sys.exit(1)
    
    if report["failures"]:
        print("FAILURES (first 10):")
        for f in report["failures"][:10]:
            print(f"  - {f['entity_type']}/{f['entity_id']}: {f['name_ar']}")
            print(f"    Reason: {f['reason']}")
        sys.exit(1)
    else:
        print("All entities found in top-K!")
        sys.exit(0)


if __name__ == "__main__":
    main()




