import asyncio, json
from pathlib import Path
from sqlalchemy import text
from apps.api.core.database import get_session

DOC_HASH='57b1bc3b0bdb89d8266fe831c9a3e7cab2958861ae8a3590a68d96a56feb90c4'
base=Path('data/derived/supplemental_ocr')/DOC_HASH
files=[
  'ea46696af060f259_physical_health_nutrition_06.json',
  '9b3b88105c6a1c18_physical_health_tadawi_07.json',
  'eb629a32ee36fd44_physical_health_fitness_08.json',
  'ec140df91afb2962_physical_health_rest_09.json',
  'a895918822bdd982_physical_health_hygiene_10.json',
  'fb5241b55d3b3547_physical_strength_endurance_01.json',
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
                       left(text_ar,160) AS head
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
                LIMIT 12
            """), {'p': f"{pref}%"})).fetchall()
            ch=(await session.execute(text("""
                SELECT chunk_id, entity_type, entity_id, chunk_type, source_anchor, left(text_ar,160) AS head
                FROM chunk
                WHERE source_anchor LIKE :p
                ORDER BY chunk_id
                LIMIT 6
            """), {'p': f"{pref}%"})).fetchall()

            print('\nFILE', fn, 'PREFIX', pref)
            print(' text_block', [dict(r._mapping) for r in tb])
            print(' evidence', [dict(r._mapping) for r in ev])
            print(' chunk', [dict(r._mapping) for r in ch])

            # Prove hierarchy edge for sub_value
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
