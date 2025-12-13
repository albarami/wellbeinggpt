import asyncio
from sqlalchemy import text
from apps.api.core.database import get_session

PREFIXES = [
    'userimg_091ee6ad24b3',  # concept C lines
    'userimg_9d955a9fcd19',  # evidence D lines
    'userimg_a3585e57a847',  # more E lines
]

async def main():
    async with get_session() as session:
        for pref in PREFIXES:
            tb = (await session.execute(text("""
                SELECT entity_type, entity_id, block_type, left(text_ar,160) AS head,
                       (source_anchor->>'source_anchor') AS anchor
                FROM text_block
                WHERE (source_anchor->>'source_anchor') LIKE :p
                ORDER BY created_at
                LIMIT 10
            """), {'p': f"{pref}%"})).fetchall()
            ev = (await session.execute(text("""
                SELECT entity_type, entity_id, evidence_type, ref_raw, ref_norm, left(text_ar,160) AS head,
                       (source_anchor->>'source_anchor') AS anchor
                FROM evidence
                WHERE (source_anchor->>'source_anchor') LIKE :p
                ORDER BY created_at
                LIMIT 10
            """), {'p': f"{pref}%"})).fetchall()
            ch = (await session.execute(text("""
                SELECT chunk_id, entity_type, entity_id, chunk_type, left(text_ar,160) AS head, source_anchor
                FROM chunk
                WHERE source_anchor LIKE :p
                ORDER BY chunk_id
                LIMIT 10
            """), {'p': f"{pref}%"})).fetchall()

            print('\nPREFIX', pref)
            print(' text_block', [dict(r._mapping) for r in tb])
            print(' evidence', [dict(r._mapping) for r in ev])
            print(' chunk', [dict(r._mapping) for r in ch])

            # If we found entity_id, prove edge linkage
            if tb:
                eid = tb[0].entity_id
                edges = (await session.execute(text("""
                    SELECT from_type, from_id, rel_type, to_type, to_id, justification
                    FROM edge
                    WHERE to_id = :eid AND rel_type IN ('CONTAINS','SUPPORTED_BY')
                    LIMIT 10
                """), {'eid': eid})).fetchall()
                print(' edges', [dict(r._mapping) for r in edges])

asyncio.run(main())
