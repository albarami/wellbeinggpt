import asyncio
from sqlalchemy import text
from apps.api.core.database import get_session

TERMS = ['Ø§Ù„ØªØ¬Ø±Ø¯', 'Ø§Ù„Ø§Ø³ØªÙ‚Ù„Ø§Ù„ÙŠØ©', 'Ø§Ù„Ø§Ø®ØªØ¨Ø§Ø±', 'Ø§Ù„Ø§Ù…ØªØ­Ø§Ù†', 'Ø§Ù„Ø§Ø¨ØªÙ„Ø§Ø¡', 'Ø§Ù„Ø­ÙƒÙ…Ø©', 'Ø±Ø§Ø¹', 'Ø¥Ù…Ø¹Ø©', 'ÙŠØ¤ØªÙŠ Ø§Ù„Ø­ÙƒÙ…Ø©']

async def q(session, sql, **params):
    rows = (await session.execute(text(sql), params)).fetchall()
    return [dict(r._mapping) for r in rows]

async def main():
    async with get_session() as session:
        # Quick counts
        counts = await q(session, """
            SELECT
              (SELECT COUNT(*) FROM pillar) AS pillars,
              (SELECT COUNT(*) FROM core_value) AS core_values,
              (SELECT COUNT(*) FROM sub_value) AS sub_values,
              (SELECT COUNT(*) FROM evidence) AS evidence,
              (SELECT COUNT(*) FROM chunk) AS chunks,
              (SELECT COUNT(*) FROM text_block) AS text_blocks
        """)
        print('counts', counts[0])

        for t in TERMS:
            sv = await q(session, """SELECT id, core_value_id, name_ar, source_anchor FROM sub_value WHERE name_ar ILIKE :p LIMIT 10""", p=f"%{t}%")
            cv = await q(session, """SELECT id, pillar_id, name_ar, source_anchor FROM core_value WHERE name_ar ILIKE :p LIMIT 10""", p=f"%{t}%")
            tb = await q(session, """SELECT entity_type, entity_id, block_type, left(text_ar,220) AS head, source_anchor FROM text_block WHERE text_ar ILIKE :p LIMIT 10""", p=f"%{t}%")
            ev = await q(session, """SELECT entity_type, entity_id, evidence_type, left(text_ar,220) AS head, ref_raw, ref_norm, source_anchor FROM evidence WHERE text_ar ILIKE :p OR ref_raw ILIKE :p LIMIT 10""", p=f"%{t}%")
            ch = await q(session, """SELECT chunk_id, entity_type, entity_id, chunk_type, left(text_ar,220) AS head, source_anchor FROM chunk WHERE text_ar ILIKE :p LIMIT 10""", p=f"%{t}%")
            if sv or cv or tb or ev or ch:
                print('\nTERM', t)
                if cv: print(' core_value', cv[:3])
                if sv: print(' sub_value', sv[:3])
                if tb: print(' text_block', tb[:3])
                if ev: print(' evidence', ev[:3])
                if ch: print(' chunk', ch[:3])

asyncio.run(main())
