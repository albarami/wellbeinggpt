import asyncio
from sqlalchemy import text
from apps.api.core.database import get_session

# Anchors from the new payloads (prefixes)
ILM_ANCHOR_PREFIX = "userimg_55a32d33429b"
SAAT_ANCHOR_PREFIX = "userimg_091ee6ad24b3"  # concept_C

async def main():
    async with get_session() as session:
        # Find sub_value rows created from user images for this batch
        sv = (await session.execute(text("""
            SELECT id, name_ar,
                   (source_anchor->>'source_anchor') AS anchor,
                   encode(convert_to(name_ar,'UTF8'),'hex') AS name_hex
            FROM sub_value
            WHERE (source_anchor->>'source_anchor') LIKE :p
            ORDER BY id
        """), {'p': f"{ILM_ANCHOR_PREFIX}%"})).fetchall()
        print('sub_values_ilm', [dict(r._mapping) for r in sv])

        # Evidence linked to the 'Ø§Ù„Ø¹Ù„Ù…' sub_value by its entity_id
        if sv:
            ilm_id = sv[0].id
            ev = (await session.execute(text("""
                SELECT evidence_type, ref_raw, ref_norm,
                       left(text_ar, 140) AS head,
                       (source_anchor->>'source_anchor') AS anchor
                FROM evidence
                WHERE entity_type='sub_value' AND entity_id=:eid
                ORDER BY created_at
            """), {'eid': ilm_id})).fetchall()
            print('evidence_for_ilm_count', len(ev))
            print('evidence_for_ilm_sample', [dict(r._mapping) for r in ev[:6]])

            # Chunk(s) for ilm
            ch = (await session.execute(text("""
                SELECT chunk_id, chunk_type, left(text_ar, 160) AS head, source_anchor
                FROM chunk
                WHERE entity_type='sub_value' AND entity_id=:eid
                ORDER BY chunk_id
                LIMIT 10
            """), {'eid': ilm_id})).fetchall()
            print('chunks_for_ilm', [dict(r._mapping) for r in ch])

        # Verify 'Ø³Ø¹Ø© Ø§Ù„Ø£ÙÙ‚' got definition/evidence chunks with anchors from its images
        saat = (await session.execute(text("""
            SELECT id, name_ar, core_value_id, (source_anchor->>'source_anchor') AS anchor
            FROM sub_value
            WHERE name_ar = 'Ø³Ø¹Ø© Ø§Ù„Ø£ÙÙ‚'
            ORDER BY created_at DESC
            LIMIT 3
        """))).fetchall()
        print('saat_rows', [dict(r._mapping) for r in saat])
        if saat:
            saat_id = saat[0].id
            # Evidence with userimg anchors (should include verses like Ø§Ù„Ø£Ù†Ø¹Ø§Ù…:104 etc)
            ev2 = (await session.execute(text("""
                SELECT evidence_type, ref_raw, ref_norm, left(text_ar, 140) AS head, (source_anchor->>'source_anchor') AS anchor
                FROM evidence
                WHERE entity_type='sub_value' AND entity_id=:eid
                ORDER BY created_at
                LIMIT 12
            """), {'eid': saat_id})).fetchall()
            print('saat_evidence_sample', [dict(r._mapping) for r in ev2])

            # Graph edge proof: core_value -> sub_value CONTAINS
            edges = (await session.execute(text("""
                SELECT from_type, from_id, rel_type, to_type, to_id, justification
                FROM edge
                WHERE rel_type='CONTAINS' AND to_type='sub_value' AND to_id=:sid
                LIMIT 5
            """), {'sid': saat_id})).fetchall()
            print('saat_edges', [dict(r._mapping) for r in edges])

asyncio.run(main())
