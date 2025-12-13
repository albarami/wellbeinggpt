import asyncio
from sqlalchemy import text
from apps.api.core.database import get_session

async def main():
    async with get_session() as session:
        row = (await session.execute(text("""
            SELECT DISTINCT e.entity_id
            FROM evidence e
            WHERE e.entity_type='sub_value'
              AND (e.source_anchor->>'source_anchor') LIKE 'userimg_9d955a9fcd19%'
            LIMIT 1
        """))).fetchone()
        sid = row.entity_id if row else None
        print('saat_id_by_anchor', sid)
        if not sid:
            return
        sv = (await session.execute(text("SELECT id, name_ar, core_value_id FROM sub_value WHERE id=:id"), {'id': sid})).fetchone()
        print('sub_value_row', dict(sv._mapping))
        edges = (await session.execute(text("""
            SELECT from_type, from_id, rel_type, to_type, to_id
            FROM edge
            WHERE rel_type='CONTAINS' AND to_type='sub_value' AND to_id=:sid
            ORDER BY from_id
        """), {'sid': sid})).fetchall()
        print('contains_edges', [dict(r._mapping) for r in edges])

asyncio.run(main())
