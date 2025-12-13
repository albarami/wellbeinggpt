import asyncio
from sqlalchemy import text
from apps.api.core.database import get_session

async def main():
    async with get_session() as session:
        # Locate 'Ø³Ø¹Ø© Ø§Ù„Ø£ÙÙ‚' id
        sid = (await session.execute(text("SELECT id FROM sub_value WHERE name_ar='Ø³Ø¹Ø© Ø§Ù„Ø£ÙÙ‚' LIMIT 1"))).scalar_one_or_none()
        print('saat_id', sid)
        if not sid:
            return
        edges = (await session.execute(text("""
            SELECT from_id, rel_type, to_id
            FROM edge
            WHERE rel_type='CONTAINS' AND to_type='sub_value' AND to_id=:sid
            ORDER BY from_id
        """), {'sid': sid})).fetchall()
        print('contains_edges', [dict(r._mapping) for r in edges])

asyncio.run(main())
