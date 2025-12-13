import json
from pathlib import Path
from apps.api.ingest.pipeline_framework import ingest_framework_docx

out_dir=Path('data/derived'); out_dir.mkdir(parents=True, exist_ok=True)
canonical_out=out_dir/'framework_2025-10_v1.suppfix10.canonical.json'
chunks_out=out_dir/'framework_2025-10_v1.suppfix10.chunks.jsonl'

summary=ingest_framework_docx(Path('docs/source/framework_2025-10_v1.docx'), canonical_out, chunks_out)
print('summary', summary)

canon=json.loads(canonical_out.read_text(encoding='utf-8'))

# Verify 'Ø§Ù„Ø­ÙƒÙ…Ø©' sub-values now include the new ones from batch3
hikmah=None
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        if cv.get('name_ar') == "\u0627\u0644\u062d\u0643\u0645\u0629":
            hikmah=cv
            pillar=p.get('name_ar')
            break

print('hikmah_pillar', pillar)
sv_names=[sv.get('name_ar') for sv in hikmah.get('sub_values', [])]
print('hikmah_sub_values', sv_names)

for target in [
    "\u0627\u0644\u0625\u0646\u062a\u0627\u062c\u064a\u0629",
    "\u0627\u0644\u062a\u0631\u0648\u064a",
    "\u0627\u0644\u062a\u0648\u0627\u0636\u0639 \u0627\u0644\u0645\u0639\u0631\u0641\u064a",
]:
    print('has', target, target in sv_names)

# Quick: show evidence counts for those subvalues
for sv in hikmah.get('sub_values', []):
    if sv.get('name_ar') in (
        "\u0627\u0644\u0625\u0646\u062a\u0627\u062c\u064a\u0629",
        "\u0627\u0644\u062a\u0631\u0648\u064a",
        "\u0627\u0644\u062a\u0648\u0627\u0636\u0639 \u0627\u0644\u0645\u0639\u0631\u0641\u064a",
    ):
        print('sub', sv.get('name_ar'), 'anchor', sv.get('source_anchor'), 'ev', len(sv.get('evidence', []) or []))
