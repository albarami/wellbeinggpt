import asyncio, json
from pathlib import Path
from sqlalchemy import text
from apps.api.core.database import get_session

DOC_HASH='57b1bc3b0bdb89d8266fe831c9a3e7cab2958861ae8a3590a68d96a56feb90c4'
base=Path('data/derived/supplemental_ocr')/DOC_HASH

# Pull anchor prefixes from the batch3 payloads directly
payload_files=[
  '02a4caa6551a0d8b_hikmah_subvalue_productivity_03_concept.json',
  '7b9488a23502ab8d_hikmah_subvalue_productivity_03_evidence.json',
  '9c2ae16f7ff5a96b_hikmah_subvalue_taanni_04_concept.json',
  'a585f467082c3a4c_hikmah_subvalue_taanni_04_more.json',
  'cd62dc8d737db99b_hikmah_subvalue_epistemic_humility_05_concept.json',
]

prefixes=[]
for fn in payload_files:
    p=base/fn
    if not p.exists():
        continue
    payload=json.loads(p.read_text(encoding='utf-8'))
    sha=str(payload.get('image_sha256',''))
    prefixes.append(f"userimg_{sha[:12]}")

async def main():
    async with get_session() as session:
        for pref in prefixes:
            # Show which entity got text_block/chunk rows for this screenshot
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

asyncio.run(main())
