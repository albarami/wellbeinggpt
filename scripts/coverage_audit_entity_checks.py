"""Coverage audit: entity/chunk/edge checks.

Reason: keep each script file <500 LOC.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from scripts.coverage_audit_utils import anchor_str_from_canonical, norm_ref_raw_for_match


async def audit_entities(*, session: AsyncSession, report: dict[str, Any], canonical: dict[str, Any], source_doc_id: str) -> None:
    """Populate report counts/missing lists for entities, chunks, embeddings, and basic graph edges."""
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
            report["missing"]["entities"].append({"type": "pillar", "name_ar": p["name_ar"], "source_doc_id": source_doc_id})
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
                report["missing"]["entities"].append({"type": "core_value", "name_ar": cv["name_ar"], "pillar": p["name_ar"]})
                cv_db_id = ""

            if cv.get("definition") and (cv["definition"].get("text_ar") or "").strip():
                report["counts"]["definition_blocks_expected"] += 1
                def_anchor = anchor_str_from_canonical(cv["definition"].get("source_anchor"))
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
                        {"entity_type": "core_value", "entity_id": cv_db_id, "name_ar": cv["name_ar"], "source_anchor": def_anchor}
                    )

            for ev in cv.get("evidence", []) or []:
                report["counts"]["evidence_blocks_expected"] += 1
                ev_anchor = anchor_str_from_canonical(ev.get("source_anchor"))
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
                        {"sd": source_doc_id, "eid": cv_db_id, "t": ev.get("evidence_type"), "text_ar": ev_hash, "a": ev_anchor},
                    )
                ).fetchall()

                want_ref = str(ev.get("ref_raw", "") or "")
                found = False
                if cand:
                    if (ev.get("evidence_type") or "") == "hadith":
                        want = norm_ref_raw_for_match(want_ref)
                        found = any(norm_ref_raw_for_match(str(r.ref_raw or "")) == want for r in cand)
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
                        {"entity_type": "core_value", "entity_id": cv_db_id, "chunk_type": "evidence", "source_anchor": ev_anchor, "ref_raw": (ev.get("ref_raw", "") or "")[:150]}
                    )

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
                            {"type": "sub_value_parent_mismatch", "name_ar": sv["name_ar"], "expected_core_value_id": cv_db_id, "found_core_value_id": str(sv_row.core_value_id)}
                        )
                else:
                    report["missing"]["entities"].append({"type": "sub_value", "name_ar": sv["name_ar"], "core_value": cv["name_ar"]})
                    sv_db_id = ""

                if sv.get("definition") and (sv["definition"].get("text_ar") or "").strip():
                    report["counts"]["definition_blocks_expected"] += 1
                    def_anchor = anchor_str_from_canonical(sv["definition"].get("source_anchor"))
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
                            {"entity_type": "sub_value", "entity_id": sv_db_id, "name_ar": sv["name_ar"], "source_anchor": def_anchor}
                        )

                for ev in sv.get("evidence", []) or []:
                    report["counts"]["evidence_blocks_expected"] += 1
                    ev_anchor = anchor_str_from_canonical(ev.get("source_anchor"))
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
                            {"sd": source_doc_id, "eid": sv_db_id, "t": ev.get("evidence_type"), "text_ar": ev_hash, "a": ev_anchor},
                        )
                    ).fetchall()

                    want_ref = str(ev.get("ref_raw", "") or "")
                    found = False
                    if cand:
                        if (ev.get("evidence_type") or "") == "hadith":
                            want = norm_ref_raw_for_match(want_ref)
                            found = any(norm_ref_raw_for_match(str(r.ref_raw or "")) == want for r in cand)
                        else:
                            found = any(str(r.ref_raw or "") == want_ref for r in cand)

                    if found:
                        report["counts"]["evidence_blocks_found"] += 1
                    else:
                        report["missing"]["evidence"].append(
                            {"entity_type": "sub_value", "entity_id": sv_db_id, "name_ar": sv.get("name_ar"), "evidence_type": ev.get("evidence_type"), "ref_raw": (ev.get("ref_raw", "") or "")[:150], "source_anchor": ev_anchor}
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
                            {"entity_type": "sub_value", "entity_id": sv_db_id, "chunk_type": "evidence", "source_anchor": ev_anchor, "ref_raw": (ev.get("ref_raw", "") or "")[:150]}
                        )

    chunk_rows = (await session.execute(text("SELECT chunk_id FROM chunk WHERE source_doc_id = :sd"), {"sd": source_doc_id})).fetchall()
    report["counts"]["chunks_expected"] = len(chunk_rows)
    report["counts"]["chunks_found"] = len(chunk_rows)
    report["counts"]["embeddings_expected"] = len(chunk_rows)

    for c in chunk_rows:
        emb_row = (await session.execute(text("SELECT id FROM embedding WHERE chunk_id = :cid"), {"cid": c.chunk_id})).fetchone()
        if emb_row:
            report["counts"]["embeddings_found"] += 1
        else:
            report["missing"]["embeddings"].append({"chunk_id": str(c.chunk_id)})

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

    entities_with_evidence = (
        await session.execute(text("SELECT DISTINCT entity_type, entity_id FROM evidence WHERE source_doc_id = :sd"), {"sd": source_doc_id})
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
            report["missing"]["graph_edges"].append({"expected_rel": "SUPPORTED_BY", "from_type": ent.entity_type, "from_id": ent.entity_id})

