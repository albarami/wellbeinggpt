"""
Graph Integrity Report

Produces a machine-checkable report for:
- Parent-child constraints (pillar->core_value, core_value->sub_value)
- Evidence linkage (SUPPORTED_BY edges for entities that have evidence)
- Orphan edges (edges pointing to non-existent nodes)

Usage:
    python scripts/graph_integrity_report.py

Env vars required:
    DATABASE_URL
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import text

from apps.api.core.database import get_session


def _require_env(name: str) -> str:
    val = (os.getenv(name) or "").strip()
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


async def run_graph_integrity_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "counts": {},
        "edge_counts": {},
        "issues": {
            "core_values_missing_parent": [],
            "sub_values_missing_parent": [],
            "entities_with_evidence_missing_supported_by": [],
            "orphan_edges": [],
        },
        "status": "pending",
    }

    async with get_session() as session:
        # Basic counts
        for tbl in ["pillar", "core_value", "sub_value", "evidence", "chunk", "edge"]:
            c = (await session.execute(text(f"SELECT COUNT(*) AS c FROM {tbl}"))).scalar_one()
            report["counts"][tbl] = int(c)

        # Edge counts by type (approved)
        edge_rows = (
            await session.execute(
                text(
                    """
                    SELECT rel_type, COUNT(*) AS cnt
                    FROM edge
                    WHERE status = 'approved'
                    GROUP BY rel_type
                    ORDER BY rel_type
                    """
                )
            )
        ).fetchall()
        for r in edge_rows:
            report["edge_counts"][r.rel_type] = int(r.cnt)

        # Parent-child constraints
        cv_bad = (
            await session.execute(
                text(
                    """
                    SELECT cv.id, cv.name_ar, cv.pillar_id
                    FROM core_value cv
                    LEFT JOIN pillar p ON p.id = cv.pillar_id
                    WHERE cv.pillar_id IS NULL OR p.id IS NULL
                    LIMIT 200
                    """
                )
            )
        ).fetchall()
        report["issues"]["core_values_missing_parent"] = [
            {"id": str(r.id), "name_ar": r.name_ar, "pillar_id": r.pillar_id} for r in cv_bad
        ]

        sv_bad = (
            await session.execute(
                text(
                    """
                    SELECT sv.id, sv.name_ar, sv.core_value_id
                    FROM sub_value sv
                    LEFT JOIN core_value cv ON cv.id = sv.core_value_id
                    WHERE sv.core_value_id IS NULL OR cv.id IS NULL
                    LIMIT 200
                    """
                )
            )
        ).fetchall()
        report["issues"]["sub_values_missing_parent"] = [
            {"id": str(r.id), "name_ar": r.name_ar, "core_value_id": r.core_value_id} for r in sv_bad
        ]

        # Entities with evidence must have SUPPORTED_BY edges
        ents = (
            await session.execute(
                text(
                    """
                    SELECT DISTINCT e.entity_type, e.entity_id
                    FROM evidence e
                    """
                )
            )
        ).fetchall()

        missing_supported: list[dict[str, Any]] = []
        for ent in ents:
            edge = (
                await session.execute(
                    text(
                        """
                        SELECT 1
                        FROM edge
                        WHERE status = 'approved'
                          AND rel_type = 'SUPPORTED_BY'
                          AND from_type = :et
                          AND from_id = :eid
                        LIMIT 1
                        """
                    ),
                    {"et": ent.entity_type, "eid": ent.entity_id},
                )
            ).fetchone()
            if not edge:
                missing_supported.append({"entity_type": ent.entity_type, "entity_id": ent.entity_id})
                if len(missing_supported) >= 200:
                    break
        report["issues"]["entities_with_evidence_missing_supported_by"] = missing_supported

        # Orphan edges (best-effort checks for known node types)
        # We only verify pillar/core_value/sub_value/evidence/chunk existence.
        orphan_rows = (
            await session.execute(
                text(
                    """
                    WITH e AS (
                      SELECT id, from_type, from_id, to_type, to_id, rel_type, status
                      FROM edge
                      WHERE status = 'approved'
                    )
                    SELECT e.id, e.from_type, e.from_id, e.to_type, e.to_id, e.rel_type
                    FROM e
                    LEFT JOIN pillar p1 ON (e.from_type='pillar' AND p1.id=e.from_id)
                    LEFT JOIN core_value cv1 ON (e.from_type='core_value' AND cv1.id=e.from_id)
                    LEFT JOIN sub_value sv1 ON (e.from_type='sub_value' AND sv1.id=e.from_id)
                    LEFT JOIN chunk ch1 ON (e.from_type='chunk' AND ch1.chunk_id=e.from_id)
                    LEFT JOIN evidence ev1 ON (e.from_type='evidence' AND ev1.id::text=e.from_id)
                    LEFT JOIN pillar p2 ON (e.to_type='pillar' AND p2.id=e.to_id)
                    LEFT JOIN core_value cv2 ON (e.to_type='core_value' AND cv2.id=e.to_id)
                    LEFT JOIN sub_value sv2 ON (e.to_type='sub_value' AND sv2.id=e.to_id)
                    LEFT JOIN chunk ch2 ON (e.to_type='chunk' AND ch2.chunk_id=e.to_id)
                    LEFT JOIN evidence ev2 ON (e.to_type='evidence' AND ev2.id::text=e.to_id)
                    WHERE
                      (
                        (e.from_type='pillar' AND p1.id IS NULL) OR
                        (e.from_type='core_value' AND cv1.id IS NULL) OR
                        (e.from_type='sub_value' AND sv1.id IS NULL) OR
                        (e.from_type='chunk' AND ch1.chunk_id IS NULL) OR
                        (e.from_type='evidence' AND ev1.id IS NULL) OR
                        (e.to_type='pillar' AND p2.id IS NULL) OR
                        (e.to_type='core_value' AND cv2.id IS NULL) OR
                        (e.to_type='sub_value' AND sv2.id IS NULL) OR
                        (e.to_type='chunk' AND ch2.chunk_id IS NULL) OR
                        (e.to_type='evidence' AND ev2.id IS NULL)
                      )
                    LIMIT 200
                    """
                )
            )
        ).fetchall()

        report["issues"]["orphan_edges"] = [
            {
                "id": str(r.id),
                "from_type": r.from_type,
                "from_id": r.from_id,
                "rel_type": r.rel_type,
                "to_type": r.to_type,
                "to_id": r.to_id,
            }
            for r in orphan_rows
        ]

    # Status
    total_issues = sum(len(v) for v in report["issues"].values())
    report["status"] = "complete" if total_issues == 0 else "incomplete"
    report["issues_count"] = total_issues
    return report


def main() -> None:
    load_dotenv()
    _require_env("DATABASE_URL")

    report = asyncio.run(run_graph_integrity_report())
    out = Path("graph_integrity_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out}")
    print(f"Status: {report['status']}, issues: {report['issues_count']}")


if __name__ == "__main__":
    main()

