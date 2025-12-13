import json
from pathlib import Path
from apps.api.ingest.pipeline_framework import ingest_framework_docx

out_dir=Path('data/derived'); out_dir.mkdir(parents=True, exist_ok=True)
canonical_out=out_dir/'framework_2025-10_v1.suppfix12.canonical.json'
chunks_out=out_dir/'framework_2025-10_v1.suppfix12.chunks.jsonl'

summary=ingest_framework_docx(Path('docs/source/framework_2025-10_v1.docx'), canonical_out, chunks_out)
print('summary', summary)

canon=json.loads(canonical_out.read_text(encoding='utf-8'))

# Verify Ø§Ù„Ø­ÙƒÙ…Ø© no longer has a fake sub-value like "Ù„Ø³Ø§Ù† Ø§Ù„Ø¹Ø±Ø¨"
hikmah=None
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        if cv.get('name_ar') == "\u0627\u0644\u062d\u0643\u0645\u0629":
            hikmah=cv
            break
hikmah_svs=[sv.get('name_ar') for sv in hikmah.get('sub_values', [])]
print('hikmah_sub_values', hikmah_svs)
print('has_lisan_subvalue', any('Ù„Ø³Ø§Ù† Ø§Ù„Ø¹Ø±Ø¨' in (n or '') for n in hikmah_svs))

# Verify Ø§Ù„Ø§Ø¹ØªØ¨Ø§Ø± has clean sub-values only
itibar=None
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        if cv.get('name_ar') == "\u0627\u0644\u0627\u0639\u062a\u0628\u0627\u0631":
            itibar=cv
            break
sv_names=[sv.get('name_ar') for sv in itibar.get('sub_values', [])]
print('itibar_sub_values', sv_names)
print('bad_itibar', [n for n in sv_names if n in ('Ø«Ø§Ù„Ø«Ø§','Ø§Ù„Ø¥Ø¨Ø¯Ø§Ø¹/Ø§Ù„Ø§Ø¨ØªÙƒØ§Ø±')])
