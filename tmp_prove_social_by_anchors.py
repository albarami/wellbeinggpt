import asyncio, json
from pathlib import Path
from sqlalchemy import text
from apps.api.core.database import get_session

DOC_HASH='57b1bc3b0bdb89d8266fe831c9a3e7cab2958861ae8a3590a68d96a56feb90c4'
base=Path('data/derived/supplemental_ocr')/DOC_HASH
files=[
  '90274991f702e06e_social_pillar_intro_00.json',
  'dfe437d05e40c0cb_social_core_care_structure.json',
  '4aa7fe04cdf28e77_social_core_cooperation_structure.json',
  '42e842f0b4048002_social_core_responsibility_structure.json',
  'd3f2d70fd6bf86d8_social_core_care_definition.json',
  'b81d8823f7458d47_social_care_empathy_01.json',
  '715f520ad4324f4a_social_care_mawadda_rahma_02.json',
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
                LIMIT 8
            """), {'p': f"{pref}%"})).fetchall()
            ev=(await session.execute(text("""
                SELECT entity_type, entity_id, evidence_type, ref_raw, ref_norm,
                       (source_anchor->>'source_anchor') AS anchor
                FROM evidence
                WHERE (source_anchor->>'source_anchor') LIKE :p
                ORDER BY created_at
                LIMIT 14
            """), {'p': f"{pref}%"})).fetchall()
            ch=(await session.execute(text("""
                SELECT chunk_id, entity_type, entity_id, chunk_type, source_anchor, left(text_ar,160) AS head
                FROM chunk
                WHERE source_anchor LIKE :p
                ORDER BY chunk_id
                LIMIT 8
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

            cid=None
            for r in tb:
                if r.entity_type=='core_value':
                    cid=r.entity_id
                    break
            if cid:
                cv=(await session.execute(text("SELECT id, name_ar, pillar_id FROM core_value WHERE id=:cid"), {'cid': cid})).fetchone()
                edges=(await session.execute(text("""
                    SELECT from_type, from_id, rel_type, to_type, to_id
                    FROM edge
                    WHERE rel_type='CONTAINS' AND to_type='core_value' AND to_id=:cid
                    LIMIT 5
                """), {'cid': cid})).fetchall()
                print(' core_value_row', dict(cv._mapping) if cv else None)
                print(' pillar_contains_edges', [dict(r._mapping) for r in edges])

asyncio.run(main())
