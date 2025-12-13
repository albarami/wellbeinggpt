import asyncio
from sqlalchemy import text
from apps.api.core.database import get_session

async def main():
    async with get_session() as session:
        # Find the entity id for the sub_value name
        r = (await session.execute(text("""
            SELECT id, core_value_id, source_anchor
            FROM sub_value
            WHERE name_ar = :n
            ORDER BY created_at DESC
            LIMIT 5
        """), {"n": "Ø§Ù„ØªØ¬Ø±Ø¯/Ø§Ù„Ø§Ø³ØªÙ‚Ù„Ø§Ù„ÙŠØ©"})).fetchall()
        print('sub_value_rows', [dict(x._mapping) for x in r])

        # Evidence that includes the hadith/ayah block text
        e = (await session.execute(text("""
            SELECT evidence_type, ref_raw, ref_norm, left(text_ar, 220) AS text_head, source_anchor
            FROM evidence
            WHERE text_ar ILIKE '%ÙƒÙ„%Ø±Ø§Ø¹%'
            ORDER BY created_at DESC
            LIMIT 10
        """))).fetchall()
        print('evidence_like_kullukum', [dict(x._mapping) for x in e])

        # Chunks containing the new term
        c = (await session.execute(text("""
            SELECT chunk_id, chunk_type, left(text_ar, 220) AS text_head, source_anchor
            FROM chunk
            WHERE text_ar ILIKE '%Ø§Ù„ØªØ¬Ø±Ø¯%' OR text_ar ILIKE '%Ø§Ù„Ø§Ø³ØªÙ‚Ù„Ø§Ù„ÙŠØ©%'
            ORDER BY chunk_id
            LIMIT 10
        """))).fetchall()
        print('chunks_term', [dict(x._mapping) for x in c])

asyncio.run(main())
