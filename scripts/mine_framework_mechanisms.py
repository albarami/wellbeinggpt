"""Mine grounded mechanism edges from framework chunks for World Model.

This script:
- Scans ingested framework chunks
- Extracts mechanism edges (ENABLES, REINFORCES, INHIBITS, TENSION_WITH, etc.)
- Reports total edges and cross-pillar edges separately
- Targets: ≥150 total edges, ≥40 cross-pillar edges

Run:
    python -m scripts.mine_framework_mechanisms --source-name framework_2025-10_v1
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.graph.mechanism_edge_miner import (
    LexiconEntry,
    MinerReport,
    MinerTargets,
    extract_mechanism_edges_from_chunk,
    is_cross_pillar_edge,
    upsert_mechanism_edges,
)
from eval.datasets.source_loader import load_dotenv_if_present


@dataclass
class MineResult:
    """Result of mining operation."""
    chunks_scanned: int
    candidate_edges: int
    cross_pillar_candidates: int
    inserted_edges: int
    inserted_spans: int
    report: MinerReport


async def _source_doc_id_for_pattern(*, session, source_name: str) -> str | None:
    """Find source document ID by name pattern."""
    row = (
        await session.execute(
            text("""
                SELECT id::text AS id
                FROM source_document
                WHERE file_name ILIKE :p
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"p": f"%{source_name}%"},
        )
    ).fetchone()
    return str(row.id) if row and getattr(row, "id", None) else None


async def _iter_framework_chunks(
    *, session, source_doc_id: str, limit: int | None
) -> list[dict[str, Any]]:
    """Iterate through framework chunks."""
    q = """
        SELECT chunk_id, chunk_type, entity_type, entity_id, text_ar
        FROM chunk
        WHERE source_doc_id::text = :sd
        ORDER BY chunk_id
    """
    params: dict[str, Any] = {"sd": str(source_doc_id)}
    if limit is not None and int(limit) > 0:
        q += " LIMIT :limit"
        params["limit"] = int(limit)
    res = await session.execute(text(q), params)
    return [dict(r._mapping) for r in res.fetchall()]


async def _get_pillar_labels(session) -> dict[str, str]:
    """Get pillar ID to Arabic label mapping."""
    result = await session.execute(
        text("SELECT id, name_ar FROM pillar")
    )
    return {str(r.id): str(r.name_ar) for r in result.fetchall()}


async def _build_mention_lexicon(session) -> list[LexiconEntry]:
    """Build a deterministic mention lexicon for within-pillar mining.

    This lexicon is used to detect core/sub value mentions inside sentences.
    """
    from apps.api.retrieve.normalize_ar import normalize_for_matching

    out: list[LexiconEntry] = []

    cv = await session.execute(text("SELECT id::text AS id, name_ar FROM core_value"))
    for r in cv.fetchall():
        name_norm = normalize_for_matching(str(getattr(r, "name_ar", "") or ""))
        if not name_norm:
            continue
        out.append(LexiconEntry(kind="core_value", id=str(r.id), name_norm=name_norm))

    sv = await session.execute(text("SELECT id::text AS id, name_ar FROM sub_value"))
    for r in sv.fetchall():
        name_norm = normalize_for_matching(str(getattr(r, "name_ar", "") or ""))
        if not name_norm:
            continue
        out.append(LexiconEntry(kind="sub_value", id=str(r.id), name_norm=name_norm))

    # Deterministic order
    out.sort(key=lambda e: (e.kind, e.id))
    return out


async def _count_detected_loops(session) -> int:
    """Count detected feedback loops in database."""
    result = await session.execute(
        text("SELECT COUNT(*) AS cnt FROM feedback_loop")
    )
    row = result.fetchone()
    return int(row.cnt) if row else 0


async def mine_and_insert(
    *,
    source_name: str,
    dry_run: bool,
    limit_chunks: int | None,
) -> MineResult:
    """Mine mechanism edges from framework and optionally insert to database.
    
    Args:
        source_name: Pattern to match source document file_name
        dry_run: If True, don't write to database
        limit_chunks: Optional limit on chunks to scan
        
    Returns:
        MineResult with counts and report
    """
    load_dotenv_if_present()
    
    report = MinerReport()
    
    async with get_session() as session:
        # Find source document
        sd = await _source_doc_id_for_pattern(session=session, source_name=source_name)
        if not sd:
            raise SystemExit(f"Could not find source_document matching file_name ILIKE '%{source_name}%'")
        
        # Get pillar labels
        pillar_labels = await _get_pillar_labels(session)
        lexicon = await _build_mention_lexicon(session)
        
        # Scan chunks
        chunks = await _iter_framework_chunks(
            session=session,
            source_doc_id=sd,
            limit=limit_chunks
        )
        report.chunks_scanned = len(chunks)
        
        # Extract edges from each chunk
        mined_all = []
        for c in chunks:
            txt = str(c.get("text_ar") or "")
            if not txt.strip():
                continue
            
            mined = extract_mechanism_edges_from_chunk(
                chunk_id=str(c.get("chunk_id") or ""),
                text_ar=txt,
                entity_type=str(c.get("entity_type") or ""),
                entity_id=str(c.get("entity_id") or ""),
                lexicon=lexicon,
            )
            mined_all.extend(mined)
        
        # Deduplicate by (from, to, relation)
        seen: set[tuple[str, str, str, str, str]] = set()
        unique_mined = []
        for e in mined_all:
            key = (e.from_ref_kind, e.from_ref_id, e.to_ref_kind, e.to_ref_id, e.relation_type)
            if key not in seen:
                seen.add(key)
                unique_mined.append(e)
        
        # Count by category
        cross_pillar = [e for e in unique_mined if is_cross_pillar_edge(e)]
        within_pillar = [e for e in unique_mined if not is_cross_pillar_edge(e)]
        
        report.total_edges = len(unique_mined)
        report.cross_pillar_edges = len(cross_pillar)
        report.within_pillar_edges = len(within_pillar)
        
        # Count by pillar
        for e in unique_mined:
            if e.from_ref_kind == "pillar":
                report.edges_by_pillar[e.from_ref_id] = report.edges_by_pillar.get(e.from_ref_id, 0) + 1
            if e.to_ref_kind == "pillar":
                report.edges_by_pillar[e.to_ref_id] = report.edges_by_pillar.get(e.to_ref_id, 0) + 1
        
        # Count by relation type
        for e in unique_mined:
            report.edges_by_relation_type[e.relation_type] = (
                report.edges_by_relation_type.get(e.relation_type, 0) + 1
            )
        
        if dry_run:
            return MineResult(
                chunks_scanned=report.chunks_scanned,
                candidate_edges=len(unique_mined),
                cross_pillar_candidates=len(cross_pillar),
                inserted_edges=0,
                inserted_spans=0,
                report=report,
            )
        
        # Insert edges
        summary = await upsert_mechanism_edges(
            session=session,
            mined=unique_mined,
            source_id=sd,
            pillar_labels=pillar_labels,
        )
        await session.commit()
        
        # Count loops
        report.loops_detected = await _count_detected_loops(session)
        
        return MineResult(
            chunks_scanned=report.chunks_scanned,
            candidate_edges=len(unique_mined),
            cross_pillar_candidates=len(cross_pillar),
            inserted_edges=int(summary.get("inserted_edges") or 0),
            inserted_spans=int(summary.get("inserted_spans") or 0),
            report=report,
        )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Mine mechanism edges from framework chunks for World Model"
    )
    ap.add_argument(
        "--source-name",
        default="framework_2025-10_v1",
        help="Substring match for source_document.file_name"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write to DB, just report counts"
    )
    ap.add_argument(
        "--limit-chunks",
        type=int,
        default=0,
        help="Optional limit for scanning chunks (0 = no limit)"
    )
    args = ap.parse_args()
    
    res = asyncio.run(
        mine_and_insert(
            source_name=str(args.source_name),
            dry_run=bool(args.dry_run),
            limit_chunks=(int(args.limit_chunks) if int(args.limit_chunks) > 0 else None),
        )
    )
    
    # Print report
    print(res.report.summary())
    print()
    print(f"candidate_edges={res.candidate_edges}")
    print(f"cross_pillar_candidates={res.cross_pillar_candidates}")
    if not args.dry_run:
        print(f"inserted_edges={res.inserted_edges}")
        print(f"inserted_spans={res.inserted_spans}")
    
    # Check targets
    targets = MinerTargets()
    if res.report.meets_targets(targets):
        print("\nOK: Targets met")
    else:
        print(
            f"\nFAIL: Targets NOT met (need: total>={targets.total_edges_min}, cross-pillar>={targets.cross_pillar_edges_min})"
        )


if __name__ == "__main__":
    main()
