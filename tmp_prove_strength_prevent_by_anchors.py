import asyncio, json
from pathlib import Path
from sqlalchemy import text
from apps.api.core.database import get_session

DOC_HASH='57b1bc3b0bdb89d8266fe831c9a3e7cab2958861ae8a3590a68d96a56feb90c4'
base=Path('data/derived/supplemental_ocr')/DOC_HASH
files=[
  'c3d7d1ce0bb7e767_physical_strength_vitality_02.json',
  '5569a81acace614f_physical_strength_body_balance_03.json',
  'eaef95775525b22a_physical_strength_skills_04.json',
  '6af0d7b7ac5b41fc_physical_prevent_immunity_01.json',
  'aa4865b905524660_physical_prevent_health_literacy_02.json',
  'c3c215b148921da0_physical_prevent_avoid_harm_03.json',
  '0e71f5c6eaaf718e_physical_prevent_early_screening_04.json',
]

prefixes=[]
for fn in files:
    payload=json.loads((base/fn).read_text(encoding='utf-8'))
    sha=str(payload.get('image_sha256',''))
    prefixes.append((fn, f"userimg_{sha[:12]}"))

async def main():
    async with get_session() as session:
        for fn, pref in prefixes:
            tb=(await session.execute(text("""
                SELECT entity_type, entity_id, block_type,
                       (source_anchor->>'source_anchor') AS anchor,
                       left(text_ar,140) AS head
                FROM text_block
                WHERE (source_anchor->>'source_anchor') LIKE :p
                ORDER BY created_at
                LIMIT 6
            """), {'p': f"{pref}%"})).fetchall()
            ev=(await session.execute(text("""
                SELECT entity_type, entity_id, evidence_type, ref_raw, ref_norm,
                       (source_anchor->>'source_anchor') AS anchor
                FROM evidence
                WHERE (source_anchor->>'source_anchor') LIKE :p
                ORDER BY created_at
                LIMIT 10
            """), {'p': f"{pref}%"})).fetchall()
            ch=(await session.execute(text("""
                SELECT chunk_id, entity_type, entity_id, chunk_type, source_anchor, left(text_ar,140) AS head
                FROM chunk
                WHERE source_anchor LIKE :p
                ORDER BY chunk_id
                LIMIT 6
            """), {'p': f"{pref}%"})).fetchall()

            print('\nFILE', fn, 'PREFIX', pref)
            print(' text_block', [dict(r._mapping) for r in tb])
            print(' evidence', [dict(r._mapping) for r in ev])
            print(' chunk', [dict(r._mapping) for r in ch])

            sid=None
            for r in tb:
                if r.entity_type=='sub_value':
                    sid=r.entity_id
                    break
            if not sid:
                for r in ev:
                    if r.entity_type=='sub_value':
                        sid=r.entity_id
                        break
            if sid:
                sv=(await session.execute(text("SELECT id, name_ar, core_value_id FROM sub_value WHERE id=:sid"), {'sid': sid})).fetchone()
                edges=(await session.execute(text("""
                    SELECT from_type, from_id, rel_type, to_type, to_id
                    FROM edge
                    WHERE rel_type='CONTAINS' AND to_type='sub_value' AND to_id=:sid
                    LIMIT 5
                """), {'sid': sid})).fetchall()
                print(' sub_value_row', dict(sv._mapping) if sv else None)
                print(' contains_edges', [dict(r._mapping) for r in edges])

asyncio.run(main())
