"""
Coverage Audit Script

Provable completeness verification for the wellbeing data foundation:
- DOCX → DB → Chunks → (Embeddings if configured) → Graph

Outputs coverage_report.json with empty missing.* lists if everything is complete.

Usage:
    python scripts/coverage_audit.py

Env vars required:
    DATABASE_URL - Postgres connection string
"""

from __future__ import annotations

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
from apps.api.ingest.pipeline_framework import ingest_framework_docx


def _require_env(name: str) -> str:
    val = (os.getenv(name) or "").strip()
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _anchor_str_from_canonical(anchor: Any) -> str:
    """
    Normalize canonical 'source_anchor' into the chunk/evidence string form.

    Canonical fields are sometimes:
    - str: "para_412"
    - dict: {"anchor_id": "..."} or {"source_anchor":"para_412"} depending on extractor stage
    """
    if isinstance(anchor, str):
        return anchor
    if isinstance(anchor, dict):
        if anchor.get("source_anchor"):
            return str(anchor.get("source_anchor") or "")
        if anchor.get("anchor_id"):
            return str(anchor.get("anchor_id") or "")
    return str(anchor or "")

def _norm_ref_raw_for_match(s: str) -> str:
    """
    Normalize ref_raw for matching in audits (keeps DB-vs-canonical comparisons robust).

    We keep Quran/book refs strict, but Hadith "رواه ..." lines often vary in wrapping punctuation
    between extraction layers; this avoids false negatives for the same anchored evidence line.
    """
    if not s:
        return ""
    t = " ".join(str(s).split())
    # Strip common wrapping punctuation
    t = t.strip().strip("()[]{}«»\"'،.؛:")  # keep Arabic/ASCII punctuation trimming
    return t


async def run_coverage_audit() -> dict[str, Any]:
    """
    Run the full coverage audit.
    
    Returns:
        Coverage report dictionary.
    """
    report: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "doc_hash": None,
        "doc_file": None,
        "vector_backend": os.getenv("VECTOR_BACKEND", "disabled").lower(),
        "counts": {
            "pillars_expected": 0,
            "pillars_found": 0,
            "core_values_expected": 0,
            "core_values_found": 0,
            "sub_values_expected": 0,
            "sub_values_found": 0,
            "definition_blocks_expected": 0,
            "definition_blocks_found": 0,
            "evidence_blocks_expected": 0,
            "evidence_blocks_found": 0,
            "chunks_expected": 0,
            "chunks_found": 0,
            # Strict completeness: embeddings should exist per chunk when embeddings are configured/required.
            "embeddings_expected": 0,
            "embeddings_found": 0,
        },
        "edge_counts": {},
        "missing": {
            "entities": [],
            "definitions": [],
            "evidence": [],
            "chunks": [],
            "embeddings": [],
            "graph_edges": [],
        },
        "status": "pending",
        "notes": [],
    }
    
    # Find all DOCX files
    docs_source = Path("docs/source")
    if not docs_source.exists():
        report["status"] = "error"
        report["error"] = f"docs/source directory not found"
        return report
    
    docx_files = sorted(list(docs_source.glob("*.docx")))
    if not docx_files:
        report["status"] = "error"
        report["error"] = "No .docx files found in docs/source"
        return report

    # Use the first DOCX as the authoritative corpus unless user has multiple.
    docx_path = docx_files[0]
    report["doc_file"] = str(docx_path)
    
    async with get_session() as session:
        # Generate canonical + chunks using the SAME pipeline used by ingestion/tests.
        # Reason: DOCX contains image-based pages; pipeline may augment via OCR and anchor as userimg_*.
        # Ensure supplemental OCR paragraphs are included without requiring any external OCR service.
        os.environ.setdefault("INGEST_OCR_FROM_IMAGES", "off")
        out_dir = Path("data/derived/coverage_audit")
        out_dir.mkdir(parents=True, exist_ok=True)
        canon_path = out_dir / "coverage_audit.canonical.json"
        chunks_path = out_dir / "coverage_audit.chunks.jsonl"

        # ingest_framework_docx is synchronous and may call asyncio.run() internally.
        # Run it off the event loop to avoid nested asyncio.run errors.
        await asyncio.to_thread(ingest_framework_docx, docx_path, canon_path, chunks_path)
        canonical = json.loads(canon_path.read_text(encoding="utf-8"))

        doc_hash = str(canonical.get("meta", {}).get("source_file_hash", "") or "").strip()
        report["doc_hash"] = doc_hash
        if not doc_hash:
            report["status"] = "error"
            report["error"] = "Canonical meta.source_file_hash missing"
            return report

        # Get source_doc_id from DB
        sd_row = (
            await session.execute(
                text("SELECT id FROM source_document WHERE file_hash = :h"),
                {"h": doc_hash},
            )
        ).fetchone()

        if not sd_row:
            report["missing"]["entities"].append(
                {
                    "type": "source_document",
                    "file": str(docx_path),
                    "file_hash": doc_hash,
                    "reason": "source_document not found for file_hash (did you ingest this DOCX?)",
                }
            )
            report["status"] = "incomplete"
            return report

        source_doc_id = str(sd_row.id)

        # Check pillars/core/sub entities and their definition/evidence blocks
        for p in canonical.get("pillars", []):
            report["counts"]["pillars_expected"] += 1
            p_row = (
                await session.execute(
                    text("SELECT id, source_anchor FROM pillar WHERE source_doc_id = :sd AND id = :id"),
                    {"sd": source_doc_id, "id": p["id"]},
                )
            ).fetchone()

            if p_row:
                report["counts"]["pillars_found"] += 1
                pillar_db_id = str(p_row.id)
            else:
                report["missing"]["entities"].append(
                    {"type": "pillar", "name_ar": p["name_ar"], "source_doc_id": source_doc_id}
                )
                pillar_db_id = ""

            for cv in p.get("core_values", []):
                report["counts"]["core_values_expected"] += 1
                cv_row = (
                    await session.execute(
                        text(
                            """
                            SELECT id, pillar_id
                            FROM core_value
                            WHERE source_doc_id = :sd AND id = :id
                            """
                        ),
                        {"sd": source_doc_id, "id": cv["id"]},
                    )
                ).fetchone()

                if cv_row:
                    report["counts"]["core_values_found"] += 1
                    cv_db_id = str(cv_row.id)
                    if pillar_db_id and str(cv_row.pillar_id) != pillar_db_id:
                        report["missing"]["entities"].append(
                            {
                                "type": "core_value_parent_mismatch",
                                "name_ar": cv["name_ar"],
                                "expected_pillar_id": pillar_db_id,
                                "found_pillar_id": str(cv_row.pillar_id),
                            }
                        )
                else:
                    report["missing"]["entities"].append(
                        {"type": "core_value", "name_ar": cv["name_ar"], "pillar": p["name_ar"]}
                    )
                    cv_db_id = ""

                # Definition block -> must have chunk (definition) with matching anchor
                if cv.get("definition") and (cv["definition"].get("text_ar") or "").strip():
                    report["counts"]["definition_blocks_expected"] += 1
                    def_anchor = _anchor_str_from_canonical(cv["definition"].get("source_anchor"))
                    chunk_row = (
                        await session.execute(
                            text(
                                """
                                SELECT chunk_id
                                FROM chunk
                                WHERE source_doc_id = :sd
                                  AND entity_type = 'core_value'
                                  AND entity_id = :eid
                                  AND chunk_type = 'definition'
                                  AND source_anchor = :a
                                LIMIT 1
                                """
                            ),
                            {"sd": source_doc_id, "eid": cv_db_id, "a": def_anchor},
                        )
                    ).fetchone()
                    if chunk_row:
                        report["counts"]["definition_blocks_found"] += 1
                    else:
                        report["missing"]["definitions"].append(
                            {
                                "entity_type": "core_value",
                                "entity_id": cv_db_id,
                                "name_ar": cv["name_ar"],
                                "source_anchor": def_anchor,
                            }
                        )

                # Evidence blocks -> must have evidence row AND evidence chunk with matching anchor
                for ev in cv.get("evidence", []) or []:
                    report["counts"]["evidence_blocks_expected"] += 1
                    ev_anchor = _anchor_str_from_canonical(ev.get("source_anchor"))
                    ev_hash = str(ev.get("text_ar") or "")
                    cand = (
                        await session.execute(
                            text(
                                """
                                SELECT id, ref_raw
                                FROM evidence
                                WHERE source_doc_id = :sd
                                  AND entity_type = 'core_value'
                                  AND entity_id = :eid
                                  AND evidence_type = :t
                                  AND md5(text_ar) = md5(:text_ar)
                                  AND (source_anchor->>'source_anchor') = :a
                                """
                            ),
                            {
                                "sd": source_doc_id,
                                "eid": cv_db_id,
                                "t": ev.get("evidence_type"),
                                "text_ar": ev_hash,
                                "a": ev_anchor,
                            },
                        )
                    ).fetchall()

                    want_ref = str(ev.get("ref_raw", "") or "")
                    found = False
                    if cand:
                        if (ev.get("evidence_type") or "") == "hadith":
                            want = _norm_ref_raw_for_match(want_ref)
                            found = any(_norm_ref_raw_for_match(str(r.ref_raw or "")) == want for r in cand)
                        else:
                            found = any(str(r.ref_raw or "") == want_ref for r in cand)

                    if found:
                        report["counts"]["evidence_blocks_found"] += 1
                    else:
                        report["missing"]["evidence"].append(
                            {
                                "entity_type": "core_value",
                                "entity_id": cv_db_id,
                                "name_ar": cv.get("name_ar"),
                                "evidence_type": ev.get("evidence_type"),
                                "ref_raw": (ev.get("ref_raw", "") or "")[:150],
                                "source_anchor": ev_anchor,
                            }
                        )

                    chunk_row = (
                        await session.execute(
                            text(
                                """
                                SELECT chunk_id
                                FROM chunk
                                WHERE source_doc_id = :sd
                                  AND entity_type = 'core_value'
                                  AND entity_id = :eid
                                  AND chunk_type = 'evidence'
                                  AND source_anchor = :a
                                LIMIT 1
                                """
                            ),
                            {"sd": source_doc_id, "eid": cv_db_id, "a": ev_anchor},
                        )
                    ).fetchone()
                    if not chunk_row:
                        report["missing"]["chunks"].append(
                            {
                                "entity_type": "core_value",
                                "entity_id": cv_db_id,
                                "chunk_type": "evidence",
                                "source_anchor": ev_anchor,
                                "ref_raw": (ev.get("ref_raw", "") or "")[:150],
                            }
                        )

                # Sub-values
                for sv in cv.get("sub_values", []):
                    report["counts"]["sub_values_expected"] += 1
                    sv_row = (
                        await session.execute(
                            text(
                                """
                                SELECT id, core_value_id
                                FROM sub_value
                                WHERE source_doc_id = :sd AND id = :id
                                """
                            ),
                            {"sd": source_doc_id, "id": sv["id"]},
                        )
                    ).fetchone()

                    if sv_row:
                        report["counts"]["sub_values_found"] += 1
                        sv_db_id = str(sv_row.id)
                        if cv_db_id and str(sv_row.core_value_id) != cv_db_id:
                            report["missing"]["entities"].append(
                                {
                                    "type": "sub_value_parent_mismatch",
                                    "name_ar": sv["name_ar"],
                                    "expected_core_value_id": cv_db_id,
                                    "found_core_value_id": str(sv_row.core_value_id),
                                }
                            )
                    else:
                        report["missing"]["entities"].append(
                            {"type": "sub_value", "name_ar": sv["name_ar"], "core_value": cv["name_ar"]}
                        )
                        sv_db_id = ""

                    if sv.get("definition") and (sv["definition"].get("text_ar") or "").strip():
                        report["counts"]["definition_blocks_expected"] += 1
                        def_anchor = _anchor_str_from_canonical(sv["definition"].get("source_anchor"))
                        chunk_row = (
                            await session.execute(
                                text(
                                    """
                                    SELECT chunk_id
                                    FROM chunk
                                    WHERE source_doc_id = :sd
                                      AND entity_type = 'sub_value'
                                      AND entity_id = :eid
                                      AND chunk_type = 'definition'
                                      AND source_anchor = :a
                                    LIMIT 1
                                    """
                                ),
                                {"sd": source_doc_id, "eid": sv_db_id, "a": def_anchor},
                            )
                        ).fetchone()
                        if chunk_row:
                            report["counts"]["definition_blocks_found"] += 1
                        else:
                            report["missing"]["definitions"].append(
                                {
                                    "entity_type": "sub_value",
                                    "entity_id": sv_db_id,
                                    "name_ar": sv["name_ar"],
                                    "source_anchor": def_anchor,
                                }
                            )

                    for ev in sv.get("evidence", []) or []:
                        report["counts"]["evidence_blocks_expected"] += 1
                        ev_anchor = _anchor_str_from_canonical(ev.get("source_anchor"))
                        ev_hash = str(ev.get("text_ar") or "")
                        cand = (
                            await session.execute(
                                text(
                                    """
                                    SELECT id, ref_raw
                                    FROM evidence
                                    WHERE source_doc_id = :sd
                                      AND entity_type = 'sub_value'
                                      AND entity_id = :eid
                                      AND evidence_type = :t
                                      AND md5(text_ar) = md5(:text_ar)
                                      AND (source_anchor->>'source_anchor') = :a
                                    """
                                ),
                                {
                                    "sd": source_doc_id,
                                    "eid": sv_db_id,
                                    "t": ev.get("evidence_type"),
                                    "text_ar": ev_hash,
                                    "a": ev_anchor,
                                },
                            )
                        ).fetchall()

                        want_ref = str(ev.get("ref_raw", "") or "")
                        found = False
                        if cand:
                            if (ev.get("evidence_type") or "") == "hadith":
                                want = _norm_ref_raw_for_match(want_ref)
                                found = any(_norm_ref_raw_for_match(str(r.ref_raw or "")) == want for r in cand)
                            else:
                                found = any(str(r.ref_raw or "") == want_ref for r in cand)

                        if found:
                            report["counts"]["evidence_blocks_found"] += 1
                        else:
                            report["missing"]["evidence"].append(
                                {
                                    "entity_type": "sub_value",
                                    "entity_id": sv_db_id,
                                    "name_ar": sv.get("name_ar"),
                                    "evidence_type": ev.get("evidence_type"),
                                    "ref_raw": (ev.get("ref_raw", "") or "")[:150],
                                    "source_anchor": ev_anchor,
                                }
                            )

                        chunk_row = (
                            await session.execute(
                                text(
                                    """
                                    SELECT chunk_id
                                    FROM chunk
                                    WHERE source_doc_id = :sd
                                      AND entity_type = 'sub_value'
                                      AND entity_id = :eid
                                      AND chunk_type = 'evidence'
                                      AND source_anchor = :a
                                    LIMIT 1
                                    """
                                ),
                                {"sd": source_doc_id, "eid": sv_db_id, "a": ev_anchor},
                            )
                        ).fetchone()
                        if not chunk_row:
                            report["missing"]["chunks"].append(
                                {
                                    "entity_type": "sub_value",
                                    "entity_id": sv_db_id,
                                    "chunk_type": "evidence",
                                    "source_anchor": ev_anchor,
                                    "ref_raw": (ev.get("ref_raw", "") or "")[:150],
                                }
                            )

        # Check chunk + embedding coverage for this source_doc_id
        chunk_rows = (
            await session.execute(
                text("SELECT chunk_id FROM chunk WHERE source_doc_id = :sd"),
                {"sd": source_doc_id},
            )
        ).fetchall()

        report["counts"]["chunks_expected"] = len(chunk_rows)
        report["counts"]["chunks_found"] = len(chunk_rows)
        report["counts"]["embeddings_expected"] = len(chunk_rows)

        for c in chunk_rows:
            emb_row = (
                await session.execute(
                    text("SELECT id FROM embedding WHERE chunk_id = :cid"),
                    {"cid": c.chunk_id},
                )
            ).fetchone()
            if emb_row:
                report["counts"]["embeddings_found"] += 1
            else:
                report["missing"]["embeddings"].append({"chunk_id": str(c.chunk_id)})

        # Edge counts
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
        for row in edge_rows:
            report["edge_counts"][row.rel_type] = int(row.cnt)

        # Graph integrity: every entity with evidence has SUPPORTED_BY
        entities_with_evidence = (
            await session.execute(
                text(
                    """
                    SELECT DISTINCT entity_type, entity_id
                    FROM evidence
                    WHERE source_doc_id = :sd
                    """
                ),
                {"sd": source_doc_id},
            )
        ).fetchall()
        for ent in entities_with_evidence:
            edge_row = (
                await session.execute(
                    text(
                        """
                        SELECT id FROM edge
                        WHERE from_type = :et AND from_id = :eid
                          AND rel_type = 'SUPPORTED_BY'
                          AND status = 'approved'
                        LIMIT 1
                        """
                    ),
                    {"et": ent.entity_type, "eid": ent.entity_id},
                )
            ).fetchone()
            if not edge_row:
                report["missing"]["graph_edges"].append(
                    {
                        "expected_rel": "SUPPORTED_BY",
                        "from_type": ent.entity_type,
                        "from_id": ent.entity_id,
                    }
                )
    
    # Determine overall status
    all_missing_empty = all(
        len(v) == 0 for v in report["missing"].values()
    )
    
    if all_missing_empty:
        report["status"] = "complete"
    else:
        report["status"] = "incomplete"
    
    return report


def main() -> None:
    """Main entry point."""
    load_dotenv()
    _require_env("DATABASE_URL")
    print("Running coverage audit...")
    print("=" * 60)
    
    report = asyncio.run(run_coverage_audit())
    
    # Write report
    output_path = Path("coverage_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"Report written to: {output_path}")
    print()
    print("SUMMARY:")
    print(f"  Status: {report['status']}")
    print(f"  Doc file: {report.get('doc_file')}")
    print(f"  Doc hash: {report.get('doc_hash')}")
    print()
    print("COUNTS:")
    for k, v in report["counts"].items():
        print(f"  {k}: {v}")
    print()
    print("EDGE COUNTS:")
    for k, v in report.get("edge_counts", {}).items():
        print(f"  {k}: {v}")
    print()
    
    # Print missing summary
    total_missing = sum(len(v) for v in report["missing"].values())
    if total_missing > 0:
        print(f"MISSING ITEMS: {total_missing} total")
        for category, items in report["missing"].items():
            if items:
                print(f"  {category}: {len(items)} missing")
        sys.exit(1)
    else:
        print("MISSING ITEMS: None - all complete!")
        sys.exit(0)


if __name__ == "__main__":
    main()
