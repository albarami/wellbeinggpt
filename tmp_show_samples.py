import asyncio
from sqlalchemy import text
from apps.api.core.database import get_session

async def main():
    async with get_session() as session:
        enc = (await session.execute(text("SHOW server_encoding"))).scalar_one()
        lc = (await session.execute(text("SHOW lc_collate"))).scalar_one()
        rows = (await session.execute(text("SELECT id, name_ar FROM sub_value ORDER BY id LIMIT 10"))).fetchall()
        print('server_encoding', enc)
        print('lc_collate', lc)
        print('sample_sub_values', [dict(r._mapping) for r in rows])

        rows2 = (await session.execute(text("SELECT id, name_ar FROM core_value ORDER BY id LIMIT 10"))).fetchall()
        print('sample_core_values', [dict(r._mapping) for r in rows2])

asyncio.run(main())
