import json
from pathlib import Path
from apps.api.ingest.pipeline_framework import ingest_framework_docx

out_dir=Path('data/derived'); out_dir.mkdir(parents=True, exist_ok=True)
canonical_out=out_dir/'framework_2025-10_v1.suppfix16.canonical.json'
chunks_out=out_dir/'framework_2025-10_v1.suppfix16.chunks.jsonl'

summary=ingest_framework_docx(Path('docs/source/framework_2025-10_v1.docx'), canonical_out, chunks_out)
print('summary', summary)

canon=json.loads(canonical_out.read_text(encoding='utf-8'))

soc=None
for p in canon.get('pillars', []):
    if p.get('name_ar') == "\u0627\u0644\u062d\u064a\u0627\u0629 \u0627\u0644\u0627\u062c\u062a\u0645\u0627\u0639\u064a\u0629":
        soc=p
        break

print('social_core_values', [cv.get('name_ar') for cv in (soc.get('core_values', []) if soc else [])])

cv_by={cv.get('name_ar'): cv for cv in (soc.get('core_values', []) if soc else [])}
for name in ["\u0627\u0644\u0631\u0639\u0627\u064a\u0629", "\u0627\u0644\u062a\u0639\u0627\u0648\u0646 \u0648\u0627\u0644\u062a\u0622\u0632\u0631", "\u0627\u0644\u0645\u0633\u0624\u0648\u0644\u064a\u0629"]:
    cv=cv_by.get(name)
    print('core', name, 'sub_values', [sv.get('name_ar') for sv in (cv.get('sub_values', []) if cv else [])])

# check specific sub-values exist
care=cv_by.get("\u0627\u0644\u0631\u0639\u0627\u064a\u0629")
need=["\u0627\u0644\u062a\u0639\u0627\u0637\u0641", "\u0627\u0644\u0645\u0648\u062f\u0629 \u0648\u0627\u0644\u0631\u062d\u0645\u0629", "\u0627\u0644\u0627\u0647\u062a\u0645\u0627\u0645", "\u0627\u0644\u0643\u0631\u0645", "\u0627\u0644\u0648\u0641\u0627\u0621"]
for t in need:
    print('has_care', t, any(sv.get('name_ar')==t for sv in (care.get('sub_values', []) if care else [])))
