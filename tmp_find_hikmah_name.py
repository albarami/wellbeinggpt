import json
from pathlib import Path

canon=json.loads(Path('data/derived/framework_2025-10_v1.suppfix2.canonical.json').read_text(encoding='utf-8'))

cvs=[]
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        cvs.append(cv.get('name_ar',''))

# Print any CV containing the Arabic root for wisdom, and the last few CVs for context.
wis=[n for n in cvs if 'Ø­ÙƒÙ…' in n]
print('core_values_with_Ø­ÙƒÙ…', wis)
print('all_core_values', cvs)
