import json
from pathlib import Path

canon=json.loads(Path('data/derived/framework_2025-10_v1.suppfix2.canonical.json').read_text(encoding='utf-8'))

hikmah=None
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        if cv.get('name_ar') == "\u0627\u0644\u062d\u0643\u0645\u0629":
            hikmah=cv
            break

print('found', bool(hikmah))
if hikmah:
    sv_names=[sv.get('name_ar') for sv in hikmah.get('sub_values', [])]
    print('sv_count', len(sv_names))
    print('sv_sample', sv_names[-15:])
    print('has_knowledge', "\u0627\u0644\u0639\u0644\u0645" in sv_names)
    print('has_saat', any("\u0633\u0639\u0629 \u0627\u0644\u0623\u0641\u0642" in (n or '') for n in sv_names))
