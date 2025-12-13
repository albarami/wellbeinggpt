import json
from pathlib import Path
from apps.api.ingest.pipeline_framework import ingest_framework_docx

out_dir=Path('data/derived'); out_dir.mkdir(parents=True, exist_ok=True)
canonical_out=out_dir/'framework_2025-10_v1.suppfix15.canonical.json'
chunks_out=out_dir/'framework_2025-10_v1.suppfix15.chunks.jsonl'

summary=ingest_framework_docx(Path('docs/source/framework_2025-10_v1.docx'), canonical_out, chunks_out)
print('summary', summary)

canon=json.loads(canonical_out.read_text(encoding='utf-8'))

# Find physical pillar, then core values Ø§Ù„Ù‚ÙˆØ© and Ø§Ù„ÙˆÙ‚Ø§ÙŠØ©
phys=None
for p in canon.get('pillars', []):
    if p.get('name_ar') == "\u0627\u0644\u062d\u064a\u0627\u0629 \u0627\u0644\u0628\u062f\u0646\u064a\u0629":
        phys=p
        break

cv_by_name={cv.get('name_ar'): cv for cv in phys.get('core_values', [])}
strength=cv_by_name.get("\u0627\u0644\u0642\u0648\u0629")
prevent=cv_by_name.get("\u0627\u0644\u0648\u0642\u0627\u064a\u0629")

print('strength_sub_values', [sv.get('name_ar') for sv in (strength.get('sub_values', []) if strength else [])])
print('prevent_sub_values', [sv.get('name_ar') for sv in (prevent.get('sub_values', []) if prevent else [])])

# Confirm key items exist
for t in ["\u0627\u0644\u062d\u064a\u0648\u064a\u0629", "\u0627\u0644\u062a\u0648\u0627\u0632\u0646 \u0627\u0644\u0628\u062f\u0646\u064a", "\u062a\u0639\u0632\u064a\u0632 \u0627\u0644\u0645\u0647\u0627\u0631\u0627\u062a \u0627\u0644\u0628\u062f\u0646\u064a\u0629"]:
    print('has_strength', t, any(sv.get('name_ar')==t for sv in (strength.get('sub_values', []) if strength else [])))

for t in ["\u062a\u0639\u0632\u064a\u0632 \u0627\u0644\u0645\u0646\u0627\u0639\u0629", "\u0627\u0644\u062b\u0642\u0627\u0641\u0629 \u0627\u0644\u0635\u062d\u064a\u0629", "\u062a\u062c\u0646\u0628 \u0627\u0644\u0636\u0631\u0631", "\u0627\u0644\u0641\u062d\u0635 \u0627\u0644\u0645\u0628\u0643\u0631"]:
    print('has_prevent', t, any(sv.get('name_ar')==t for sv in (prevent.get('sub_values', []) if prevent else [])))
