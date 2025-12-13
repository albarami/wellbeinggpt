import asyncio
from sqlalchemy import text
from apps.api.core.database import get_session

s='Ø§Ù„Ø­ÙƒÙ…Ø©'
print('py_utf8_hex', s.encode('utf-8').hex())
print('py_len', len(s), 'py_octets', len(s.encode('utf-8')))

async def main():
    async with get_session() as session:
        row = (await session.execute(text("""
            SELECT id, name_ar,
                   length(name_ar) AS char_len,
                   octet_length(name_ar) AS oct_len,
                   encode(convert_to(name_ar,'UTF8'),'hex') AS hex
            FROM core_value
            WHERE id='CV008'
        """))).fetchone()
        print('db_row', dict(row._mapping) if row else None)

        # Exact equality check
        eq = (await session.execute(text("SELECT id FROM core_value WHERE name_ar = :x"), {'x': s})).fetchall()
        print('eq_matches', [dict(r._mapping) for r in eq])

asyncio.run(main())
