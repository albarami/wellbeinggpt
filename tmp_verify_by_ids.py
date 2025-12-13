import asyncio
import json
from pathlib import Path
from sqlalchemy import text
from apps.api.core.database import get_session

CANON_PATH=Path('data/derived/framework_2025-10_v1.ocr.canonical.json')
CHUNKS_PATH=Path('data/derived/framework_2025-10_v1.ocr.chunks.jsonl')

canon=json.loads(CANON_PATH.read_text(encoding='utf-8'))

# Find IDs by scanning canonical JSON (no hardcoded Arabic literals).
sv_targets=[]
cv_targets=[]

for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        name=cv.get('name_ar','')
        if 'Ø­ÙƒÙ…Ø©' in name or 'Ø§Ù„Ø­ÙƒÙ…' in name:
            cv_targets.append((cv.get('id'), name, p.get('name_ar')))
        for sv in cv.get('sub_values', []):
            n=sv.get('name_ar','')
            if ('ØªØ¬Ø±Ø¯' in n) or ('Ø§Ø³ØªÙ‚Ù„Ø§Ù„' in n) or ('Ø§Ø®ØªØ¨Ø§Ø±' in n) or ('Ø§Ù…ØªØ­Ø§Ù†' in n) or ('Ø§Ø¨ØªÙ„Ø§Ø¡' in n):
                sv_targets.append((sv.get('id'), n, cv.get('id'), cv.get('name_ar'), p.get('name_ar')))

# Collect some chunk_ids for these entities.
chunk_ids=set()
if CHUNKS_PATH.exists():
    for line in CHUNKS_PATH.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row=json.loads(line)
        eid=row.get('entity_id')
        if any(eid==t[0] for t in sv_targets) or any(eid==t[0] for t in cv_targets):
            chunk_ids.add(row.get('chunk_id'))

async def main():
    async with get_session() as session:
        print('canonical_cv_targets', cv_targets[:10])
        print('canonical_sv_targets', sv_targets[:20])
        if cv_targets:
            ids=[t[0] for t in cv_targets]
            rows=(await session.execute(text('SELECT id, name_ar, pillar_id, source_anchor FROM core_value WHERE id = ANY(:ids)'), {'ids': ids})).fetchall()
            print('db_core_values', [dict(r._mapping) for r in rows])

        if sv_targets:
            ids=[t[0] for t in sv_targets]
            rows=(await session.execute(text('SELECT id, name_ar, core_value_id, source_anchor FROM sub_value WHERE id = ANY(:ids)'), {'ids': ids})).fetchall()
            print('db_sub_values', [dict(r._mapping) for r in rows])

            # Evidence for these sub_values
            rows=(await session.execute(text("""
                SELECT entity_id, evidence_type, ref_raw, ref_norm, left(text_ar,180) AS head, source_anchor
                FROM evidence
                WHERE entity_type='sub_value' AND entity_id = ANY(:ids)
                ORDER BY created_at DESC
                LIMIT 20
            """), {'ids': ids})).fetchall()
            print('db_evidence_for_targets', [dict(r._mapping) for r in rows])

        if chunk_ids:
            cids=sorted([c for c in chunk_ids if c])[:30]
            rows=(await session.execute(text('SELECT chunk_id, entity_type, entity_id, chunk_type, left(text_ar,180) AS head, source_anchor FROM chunk WHERE chunk_id = ANY(:cids)'), {'cids': cids})).fetchall()
            print('db_chunks_for_targets', [dict(r._mapping) for r in rows])

asyncio.run(main())
