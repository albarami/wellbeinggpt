import asyncio, json
from pathlib import Path
from sqlalchemy import text
from apps.api.core.database import get_session

DOC_HASH='57b1bc3b0bdb89d8266fe831c9a3e7cab2958861ae8a3590a68d96a56feb90c4'
base=Path('data/derived/supplemental_ocr')/DOC_HASH
files=[
  '227ce200702f227c_itibar_core_concept_00.json',
  '47873c66c5762700_itibar_core_evidence_01.json',
  '6780b37a2824cad1_itibar_sub_tadabbur_01.json',
  'a0239efddcdd926b_itibar_sub_ibdaa_02.json',
  '4a0633a3540d0e48_itibar_sub_ruya_03.json',
]

prefixes=[]
for fn in files:
    p=base/fn
    payload=json.loads(p.read_text(encoding='utf-8'))
    sha=str(payload.get('image_sha256',''))
    prefixes.append(f"userimg_{sha[:12]}")

async def main():
    async with get_session() as session:
        for pref in prefixes:
            tb=(await session.execute(text("""
                SELECT entity_type, entity_id, block_type, (source_anchor->>'source_anchor') AS anchor, left(text_ar,120) AS head
                FROM text_block
                WHERE (source_anchor->>'source_anchor') LIKE :p
                ORDER BY created_at
                LIMIT 5
            """), {'p': f"{pref}%"})).fetchall()
            ev=(await session.execute(text("""
                SELECT entity_type, entity_id, evidence_type, ref_raw, ref_norm, (source_anchor->>'source_anchor') AS anchor
                FROM evidence
                WHERE (source_anchor->>'source_anchor') LIKE :p
                ORDER BY created_at
                LIMIT 8
            """), {'p': f"{pref}%"})).fetchall()
            ch=(await session.execute(text("""
                SELECT chunk_id, entity_type, entity_id, chunk_type, source_anchor, left(text_ar,120) AS head
                FROM chunk
                WHERE source_anchor LIKE :p
                ORDER BY chunk_id
                LIMIT 5
            """), {'p': f"{pref}%"})).fetchall()

            print('\nPREFIX', pref)
            print(' text_block', [dict(r._mapping) for r in tb])
            print(' evidence', [dict(r._mapping) for r in ev])
            print(' chunk', [dict(r._mapping) for r in ch])

            # If evidence exists, show CONTAINS edge for sub_value targets
            if tb and tb[0].entity_type == 'sub_value':
                sid = tb[0].entity_id
                edges = (await session.execute(text("""
                    SELECT from_type, from_id, rel_type, to_type, to_id
                    FROM edge
                    WHERE rel_type='CONTAINS' AND to_type='sub_value' AND to_id=:sid
                    LIMIT 5
                """), {'sid': sid})).fetchall()
                print(' contains_edges', [dict(r._mapping) for r in edges])

asyncio.run(main())
