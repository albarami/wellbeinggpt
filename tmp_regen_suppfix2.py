import json
from pathlib import Path
from apps.api.ingest.pipeline_framework import ingest_framework_docx

out_dir=Path('data/derived')
out_dir.mkdir(parents=True, exist_ok=True)
canonical_out=out_dir/'framework_2025-10_v1.suppfix2.canonical.json'
chunks_out=out_dir/'framework_2025-10_v1.suppfix2.chunks.jsonl'

summary=ingest_framework_docx(Path('docs/source/framework_2025-10_v1.docx'), canonical_out, chunks_out)
print('summary', summary)

canon=json.loads(canonical_out.read_text(encoding='utf-8'))

# Find the CV 'Ø§Ù„Ø­ÙƒÙ…Ø©' and print its sub-values list (names only)
hikmah=None
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        if cv.get('name_ar') == 'Ø§Ù„Ø­ÙƒÙ…Ø©':
            hikmah=cv
            break

if not hikmah:
    raise SystemExit('no hikmah found')

sv_names=[sv.get('name_ar') for sv in hikmah.get('sub_values', [])]
print('hikmah_sub_values_count', len(sv_names))
print('hikmah_sub_values_slashy', [n for n in sv_names if '/' in (n or '')])
print('has_knowledge', any(n=='Ø§Ù„Ø¹Ù„Ù…' for n in sv_names))
print('has_saat_ufuq', any('Ø³Ø¹Ø© Ø§Ù„Ø£ÙÙ‚' in (n or '') for n in sv_names))
