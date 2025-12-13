import json
from pathlib import Path
from apps.api.ingest.pipeline_framework import ingest_framework_docx

out_dir=Path('data/derived'); out_dir.mkdir(parents=True, exist_ok=True)
canonical_out=out_dir/'framework_2025-10_v1.suppfix11.canonical.json'
chunks_out=out_dir/'framework_2025-10_v1.suppfix11.chunks.jsonl'

summary=ingest_framework_docx(Path('docs/source/framework_2025-10_v1.docx'), canonical_out, chunks_out)
print('summary', summary)

canon=json.loads(canonical_out.read_text(encoding='utf-8'))

# Find core value Ø§Ù„Ø§Ø¹ØªØ¨Ø§Ø± and show its sub-values
itibar=None
pillar=None
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        if cv.get('name_ar') == "\u0627\u0644\u0627\u0639\u062a\u0628\u0627\u0631":
            itibar=cv
            pillar=p.get('name_ar')
            break

if not itibar:
    raise SystemExit('core_value Ø§Ù„Ø§Ø¹ØªØ¨Ø§Ø± not found')

sv_names=[sv.get('name_ar') for sv in itibar.get('sub_values', [])]
print('itibar_pillar', pillar)
print('itibar_sub_values', sv_names)

for target in ["\u0627\u0644\u062a\u062f\u0628\u0631", "\u0627\u0644\u0625\u0628\u062f\u0627\u0639/\u0627\u0644\u0627\u0628\u062a\u0643\u0627\u0631", "\u0627\u0644\u0631\u0624\u064a\u0629"]:
    print('has', target, target in sv_names)

# Show evidence counts
print('core_evidence_count', len(itibar.get('evidence', []) or []))
for sv in itibar.get('sub_values', []):
    if sv.get('name_ar') in ("\u0627\u0644\u062a\u062f\u0628\u0631", "\u0627\u0644\u0625\u0628\u062f\u0627\u0639/\u0627\u0644\u0627\u0628\u062a\u0643\u0627\u0631", "\u0627\u0644\u0631\u0624\u064a\u0629"):
        print('sub', sv.get('name_ar'), 'anchor', sv.get('source_anchor'), 'ev', len(sv.get('evidence', []) or []))
