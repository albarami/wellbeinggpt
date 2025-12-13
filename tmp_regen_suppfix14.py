import json
from pathlib import Path
from apps.api.ingest.pipeline_framework import ingest_framework_docx

out_dir=Path('data/derived'); out_dir.mkdir(parents=True, exist_ok=True)
canonical_out=out_dir/'framework_2025-10_v1.suppfix14.canonical.json'
chunks_out=out_dir/'framework_2025-10_v1.suppfix14.chunks.jsonl'

summary=ingest_framework_docx(Path('docs/source/framework_2025-10_v1.docx'), canonical_out, chunks_out)
print('summary', summary)

canon=json.loads(canonical_out.read_text(encoding='utf-8'))

# Verify physical pillar exists and core values include Ø§Ù„ØµØ­Ø© and Ø§Ù„Ù‚ÙˆØ©
phys=None
for p in canon.get('pillars', []):
    if p.get('name_ar') == "\u0627\u0644\u062d\u064a\u0627\u0629 \u0627\u0644\u0628\u062f\u0646\u064a\u0629":
        phys=p
        break

print('found_physical', bool(phys))
cv_names=[cv.get('name_ar') for cv in phys.get('core_values',[])]
print('physical_core_values', cv_names)

# Check that health subvalues are present
health=None
for cv in phys.get('core_values', []):
    if cv.get('name_ar') == "\u0627\u0644\u0635\u062d\u0629":
        health=cv
strength=None
for cv in phys.get('core_values', []):
    if cv.get('name_ar') == "\u0627\u0644\u0642\u0648\u0629":
        strength=cv

if health:
    sv_names=[sv.get('name_ar') for sv in health.get('sub_values', [])]
    print('health_sub_values', sv_names)
    for t in [
        "\u0627\u0644\u062a\u063a\u0630\u064a\u0629 \u0627\u0644\u0633\u0644\u064a\u0645\u0629",
        "\u0627\u0644\u062a\u062f\u0627\u0648\u064a",
        "\u0627\u0644\u0644\u064a\u0627\u0642\u0629",
        "\u0627\u0644\u0631\u0627\u062d\u0629 \u0627\u0644\u0643\u0627\u0641\u064a\u0629",
        "\u0627\u0644\u0646\u0638\u0627\u0641\u0629 \u0627\u0644\u0634\u062e\u0635\u064a\u0629",
    ]:
        print('has', t, t in sv_names)

if strength:
    sv_names=[sv.get('name_ar') for sv in strength.get('sub_values', [])]
    print('strength_sub_values', sv_names)
    print('has_endurance', "\u0627\u0644\u0642\u062f\u0631\u0629 \u0639\u0644\u0649 \u0627\u0644\u062a\u062d\u0645\u0644" in sv_names)

# Ensure anchors exist
s=json.dumps(canon, ensure_ascii=False)
print('has_physical_userimg', 'userimg_' in s)
