import asyncio
from sqlalchemy import text
from apps.api.core.database import get_session

async def main():
    async with get_session() as session:
        sv = (await session.execute(text("""
            SELECT id, name_ar, core_value_id, source_anchor
            FROM sub_value
            WHERE source_anchor::text LIKE '%userimg_%'
            ORDER BY id
        """))).fetchall()
        cv = (await session.execute(text("""
            SELECT id, name_ar, pillar_id, source_anchor
            FROM core_value
            WHERE source_anchor::text LIKE '%userimg_%'
            ORDER BY id
        """))).fetchall()
        tb = (await session.execute(text("""
            SELECT entity_type, entity_id, block_type, left(text_ar,160) AS head, source_anchor
            FROM text_block
            WHERE source_anchor::text LIKE '%userimg_%'
            ORDER BY created_at DESC
            LIMIT 50
        """))).fetchall()
        ev = (await session.execute(text("""
            SELECT entity_type, entity_id, evidence_type, left(text_ar,160) AS head, ref_raw, source_anchor
            FROM evidence
            WHERE source_anchor::text LIKE '%userimg_%'
            ORDER BY created_at DESC
            LIMIT 50
        """))).fetchall()
        ch = (await session.execute(text("""
            SELECT chunk_id, entity_type, entity_id, chunk_type, left(text_ar,160) AS head, source_anchor
            FROM chunk
            WHERE source_anchor LIKE 'userimg_%'
            ORDER BY chunk_id
            LIMIT 50
        """))).fetchall()

        print('sub_values_userimg', [dict(r._mapping) for r in sv])
        print('core_values_userimg', [dict(r._mapping) for r in cv])
        print('text_blocks_userimg', [dict(r._mapping) for r in tb])
        print('evidence_userimg', [dict(r._mapping) for r in ev])
        print('chunks_userimg', [dict(r._mapping) for r in ch])

asyncio.run(main())
