import asyncio
from sqlalchemy import text
from apps.api.core.database import get_session

async def main():
    async with get_session() as session:
        like = (await session.execute(text("SELECT id, name_ar FROM core_value WHERE name_ar LIKE '%Ø§Ù„Ø­ÙƒÙ…Ø©%'"))).fetchall()
        ilike = (await session.execute(text("SELECT id, name_ar FROM core_value WHERE name_ar ILIKE '%Ø§Ù„Ø­ÙƒÙ…Ø©%'"))).fetchall()
        pos = (await session.execute(text("SELECT id, name_ar FROM core_value WHERE position('Ø§Ù„Ø­ÙƒÙ…Ø©' in name_ar) > 0"))).fetchall()
        print('LIKE', [dict(r._mapping) for r in like])
        print('ILIKE', [dict(r._mapping) for r in ilike])
        print('POSITION', [dict(r._mapping) for r in pos])

asyncio.run(main())
