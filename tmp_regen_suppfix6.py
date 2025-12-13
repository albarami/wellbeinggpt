import json
from pathlib import Path
from apps.api.ingest.pipeline_framework import ingest_framework_docx

out_dir=Path('data/derived')
out_dir.mkdir(parents=True, exist_ok=True)
canonical_out=out_dir/'framework_2025-10_v1.suppfix6.canonical.json'
chunks_out=out_dir/'framework_2025-10_v1.suppfix6.chunks.jsonl'

summary=ingest_framework_docx(Path('docs/source/framework_2025-10_v1.docx'), canonical_out, chunks_out)
print('summary', summary)

canon=json.loads(canonical_out.read_text(encoding='utf-8'))

# Find Ø§Ù„Ø­ÙƒÙ…Ø© and print sub-values
hikmah=None
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        if cv.get('name_ar') == "\u0627\u0644\u062d\u0643\u0645\u0629":
            hikmah=cv
            break

sv_names=[sv.get('name_ar') for sv in hikmah.get('sub_values', [])]
print('hikmah_sub_values', sv_names)
print('has_ilm', "\u0627\u0644\u0639\u0644\u0645" in sv_names)

# If present, show evidence count for Ø§Ù„Ø¹Ù„Ù…
for sv in hikmah.get('sub_values', []):
    if sv.get('name_ar') == "\u0627\u0644\u0639\u0644\u0645":
        print('ilm_anchor', sv.get('source_anchor'))
        print('ilm_def_head', (sv.get('definition') or {}).get('text_ar','')[:140])
        print('ilm_evidence_count', len(sv.get('evidence', []) or []))
