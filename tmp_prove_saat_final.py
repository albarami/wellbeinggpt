import asyncio
from sqlalchemy import text
from apps.api.core.database import get_session

async def main():
    async with get_session() as session:
        # Locate 'Ø³Ø¹Ø© Ø§Ù„Ø£ÙÙ‚' sub_value by looking for evidence anchors from userimg D/E
        row = (await session.execute(text("""
            SELECT DISTINCT e.entity_id
            FROM evidence e
            WHERE e.entity_type='sub_value'
              AND (e.source_anchor->>'source_anchor') LIKE 'userimg_9d955a9fcd19%'
            LIMIT 1
        """))).fetchone()
        if not row:
            print('no_saat_entity_found')
            return
        sid = row.entity_id
        sv = (await session.execute(text("""
            SELECT id, name_ar, core_value_id,
                   (source_anchor->>'source_anchor') AS anchor
            FROM sub_value
            WHERE id=:id
        """), {'id': sid})).fetchone()
        print('saat_sub_value', dict(sv._mapping))

        ev = (await session.execute(text("""
            SELECT evidence_type, ref_raw, ref_norm, (source_anchor->>'source_anchor') AS anchor
            FROM evidence
            WHERE entity_type='sub_value' AND entity_id=:id
            ORDER BY created_at
            LIMIT 25
        """), {'id': sid})).fetchall()
        print('saat_evidence_sample', [dict(r._mapping) for r in ev[:10]])

        # Graph edge proof: CV008 -> sub_value
        edges = (await session.execute(text("""
            SELECT from_type, from_id, rel_type, to_type, to_id, justification
            FROM edge
            WHERE rel_type='CONTAINS' AND to_type='sub_value' AND to_id=:id
            LIMIT 5
        """), {'id': sid})).fetchall()
        print('saat_contains_edge', [dict(r._mapping) for r in edges])

        # Chunk proof for the evidence blocks
        ch = (await session.execute(text("""
            SELECT chunk_id, chunk_type, source_anchor, left(text_ar,140) AS head
            FROM chunk
            WHERE entity_type='sub_value' AND entity_id=:id
            ORDER BY chunk_id
            LIMIT 10
        """), {'id': sid})).fetchall()
        print('saat_chunks', [dict(r._mapping) for r in ch])

asyncio.run(main())
