import json
from pathlib import Path
from apps.api.ingest.pipeline_framework import ingest_framework_docx

out_dir=Path('data/derived'); out_dir.mkdir(parents=True, exist_ok=True)
canonical_out=out_dir/'framework_2025-10_v1.suppfix13.canonical.json'
chunks_out=out_dir/'framework_2025-10_v1.suppfix13.chunks.jsonl'

summary=ingest_framework_docx(Path('docs/source/framework_2025-10_v1.docx'), canonical_out, chunks_out)
print('summary', summary)

canon=json.loads(canonical_out.read_text(encoding='utf-8'))

# Find Ø§Ù„Ø§Ø¹ØªØ¨Ø§Ø± and verify the two sub-values exist with evidence counts
itibar=None
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        if cv.get('name_ar') == "\u0627\u0644\u0627\u0639\u062a\u0628\u0627\u0631":
            itibar=cv
            break

svs=itibar.get('sub_values', [])
name_to_ev={sv.get('name_ar'): len(sv.get('evidence',[]) or []) for sv in svs}
print('itibar_sub_value_names', [sv.get('name_ar') for sv in svs])
print('ev_counts', {k: name_to_ev.get(k) for k in ["\u0627\u0644\u0648\u0639\u064a \u0628\u0627\u0644\u0633\u0646\u0646", "\u0627\u0644\u062a\u0642\u0648\u064a\u0645 \u0627\u0644\u0630\u0627\u062a\u064a"]})
