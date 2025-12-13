import json
from pathlib import Path

canon=json.loads(Path('data/derived/framework_2025-10_v1.suppfix2.canonical.json').read_text(encoding='utf-8'))

# Locate any sub-value named 'Ø§Ù„Ø¹Ù„Ù…' across entire doc
hits=[]
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        for sv in cv.get('sub_values', []):
            if sv.get('name_ar') == "\u0627\u0644\u0639\u0644\u0645":
                hits.append({'pillar': p.get('name_ar'), 'core_value': cv.get('name_ar'), 'id': sv.get('id'), 'anchor': sv.get('source_anchor')})

print('found_science_count', len(hits))
print('hits', hits)

# Also search for any sub-value where definition text contains 'Ø§Ù„Ø§Ø¹ØªÙ‚Ø§Ø¯ Ø§Ù„Ø¬Ø§Ø²Ù…'
def_hits=[]
needle = "\u0627\u0644\u0627\u0639\u062a\u0642\u0627\u062f \u0627\u0644\u062c\u0627\u0632\u0645"
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        for sv in cv.get('sub_values', []):
            d = (sv.get('definition') or {}).get('text_ar','')
            if needle in d:
                def_hits.append({'pillar': p.get('name_ar'), 'core_value': cv.get('name_ar'), 'sub_value': sv.get('name_ar'), 'id': sv.get('id')})

print('def_hits', def_hits)
