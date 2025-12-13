import json
from pathlib import Path
from apps.api.ingest.pipeline_framework import ingest_framework_docx

out_dir=Path('data/derived')
out_dir.mkdir(parents=True, exist_ok=True)
canonical_out=out_dir/'framework_2025-10_v1.suppfix9.canonical.json'
chunks_out=out_dir/'framework_2025-10_v1.suppfix9.chunks.jsonl'

summary=ingest_framework_docx(Path('docs/source/framework_2025-10_v1.docx'), canonical_out, chunks_out)
print('summary', summary)

canon=json.loads(canonical_out.read_text(encoding='utf-8'))

hikmah=None
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        if cv.get('name_ar') == "\u0627\u0644\u062d\u0643\u0645\u0629":
            hikmah=cv
            break

svs=hikmah.get('sub_values', [])
sv_names=[sv.get('name_ar') for sv in svs]
print('hikmah_sub_values', sv_names)

# Evidence should now attach to sub-value 'Ø³Ø¹Ø© Ø§Ù„Ø£ÙÙ‚'
for sv in svs:
    if sv.get('name_ar') == "\u0633\u0639\u0629 \u0627\u0644\u0623\u0641\u0642":
        ev=sv.get('evidence', []) or []
        anchors=sorted(set([e.get('source_anchor','') for e in ev if e.get('source_anchor','')]))
        print('saat_evidence_count', len(ev))
        print('saat_evidence_anchors_prefixes', sorted(set([a.split('_ln')[0] for a in anchors]))[:10])

# Ensure no mojibake 'Ã˜' sub-values exist
bad=[n for n in sv_names if n and ('Ã˜' in n or 'Ã™' in n)]
print('mojibake_subvalues', bad)
