"""
DB Inspection Utility (internal)

Usage examples:
  python scripts/inspect_db.py chunks sub_value SV055 definition
  python scripts/inspect_db.py sub_value_by_name "الحياء"

Requires:
  DATABASE_URL
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import text

from apps.api.core.database import get_session


def _require_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


async def _chunks(entity_type: str, entity_id: str, chunk_type: str) -> None:
    async with get_session() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT chunk_id, source_anchor, substring(text_ar from 1 for 120) AS t
                    FROM chunk
                    WHERE entity_type = :et AND entity_id = :eid AND chunk_type = :ct
                    ORDER BY chunk_id
                    """
                ),
                {"et": entity_type, "eid": entity_id, "ct": chunk_type},
            )
        ).fetchall()
        print(f"chunks {entity_type}/{entity_id} type={chunk_type}: {len(rows)}")
        for r in rows[:25]:
            # Row.t may conflict with SQLAlchemy internals; access by mapping key.
            print(r.chunk_id, r.source_anchor, r._mapping["t"])


async def _sub_value_by_name(name_ar: str) -> None:
    async with get_session() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT sv.id, sv.name_ar, sv.core_value_id, cv.name_ar AS core_name, cv.pillar_id
                    FROM sub_value sv
                    JOIN core_value cv ON cv.id = sv.core_value_id
                    WHERE sv.name_ar = :n
                    ORDER BY sv.id
                    """
                ),
                {"n": name_ar},
            )
        ).fetchall()
        print(f"sub_value rows for name={name_ar}: {len(rows)}")
        for r in rows:
            print(r.id, r.name_ar, "core:", r.core_value_id, r.core_name, "pillar:", r.pillar_id)

async def _evidence(entity_type: str, entity_id: str) -> None:
    async with get_session() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT id::text AS id, evidence_type, ref_raw, source_anchor->>'source_anchor' AS a,
                           substring(text_ar from 1 for 120) AS t
                    FROM evidence
                    WHERE entity_type = :et AND entity_id = :eid
                    ORDER BY evidence_type, id
                    """
                ),
                {"et": entity_type, "eid": entity_id},
            )
        ).fetchall()
        print(f"evidence rows for {entity_type}/{entity_id}: {len(rows)}")
        for r in rows[:50]:
            print(r.id, r.evidence_type, r.a, (r.ref_raw or '')[:80])


async def _evidence_find(entity_type: str, entity_id: str, anchor: str, contains: str) -> None:
    async with get_session() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT id::text AS id, evidence_type, ref_raw, source_anchor->>'source_anchor' AS a
                    FROM evidence
                    WHERE entity_type = :et AND entity_id = :eid
                      AND (source_anchor->>'source_anchor') = :a
                      AND ref_raw ILIKE '%' || :q || '%'
                    ORDER BY id
                    LIMIT 50
                    """
                ),
                {"et": entity_type, "eid": entity_id, "a": anchor, "q": contains},
            )
        ).fetchall()
        print(f"evidence_find {entity_type}/{entity_id} anchor={anchor} contains={contains}: {len(rows)}")
        for r in rows:
            print(r.id, r.evidence_type, r.a, (r.ref_raw or '')[:120])


async def _edge_count(where_sql: str, params: dict[str, Any]) -> None:
    async with get_session() as s:
        c = (
            await s.execute(
                text(f"SELECT COUNT(*) AS c FROM edge WHERE {where_sql}"),
                params,
            )
        ).scalar_one()
        print("edge count:", int(c))

def main(argv: list[str]) -> None:
    load_dotenv()
    _require_env("DATABASE_URL")

    if len(argv) < 2:
        raise SystemExit("Usage: python scripts/inspect_db.py <cmd> ...")

    cmd = argv[1]
    if cmd == "chunks":
        if len(argv) != 5:
            raise SystemExit("Usage: python scripts/inspect_db.py chunks <entity_type> <entity_id> <chunk_type>")
        asyncio.run(_chunks(argv[2], argv[3], argv[4]))
        return
    if cmd == "sub_value_by_name":
        if len(argv) != 3:
            raise SystemExit("Usage: python scripts/inspect_db.py sub_value_by_name \"<name_ar>\"")
        asyncio.run(_sub_value_by_name(argv[2]))
        return
    if cmd == "evidence":
        if len(argv) != 4:
            raise SystemExit("Usage: python scripts/inspect_db.py evidence <entity_type> <entity_id>")
        asyncio.run(_evidence(argv[2], argv[3]))
        return
    if cmd == "evidence_find":
        if len(argv) != 6:
            raise SystemExit("Usage: python scripts/inspect_db.py evidence_find <entity_type> <entity_id> <anchor> <contains>")
        asyncio.run(_evidence_find(argv[2], argv[3], argv[4], argv[5]))
        return
    if cmd == "edge_count":
        if len(argv) < 3:
            raise SystemExit("Usage: python scripts/inspect_db.py edge_count \"<where_sql>\" [k=v ...]")
        where_sql = argv[2]
        params: dict[str, Any] = {}
        for kv in argv[3:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                params[k] = v
        asyncio.run(_edge_count(where_sql, params))
        return

    raise SystemExit(f"Unknown cmd: {cmd}")


if __name__ == "__main__":
    main(sys.argv)





