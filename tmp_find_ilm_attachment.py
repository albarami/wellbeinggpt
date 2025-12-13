import json
from pathlib import Path

canon=json.loads(Path('data/derived/framework_2025-10_v1.suppfix3.canonical.json').read_text(encoding='utf-8'))
needle = "\u0627\u0644\u0627\u0639\u062a\u0642\u0627\u062f \u0627\u0644\u062c\u0627\u0632\u0645"

hits=[]
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        for sv in cv.get('sub_values', []):
            d=(sv.get('definition') or {}).get('text_ar','')
            if needle in d:
                hits.append({'pillar': p.get('name_ar'), 'core_value': cv.get('name_ar'), 'sub_value': sv.get('name_ar'), 'id': sv.get('id'), 'anchor': sv.get('source_anchor')})

print('hits', hits)

# Also: locate any chunk that begins with 'Ø§Ù„Ø¹Ù„Ù…:'
# (We scan canonical-derived chunks file)
from pathlib import Path
chunks=Path('data/derived/framework_2025-10_v1.suppfix3.chunks.jsonl')
if chunks.exists():
    import json
    for line in chunks.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row=json.loads(line)
        txt=row.get('text_ar','')
        if txt.startswith("\u0627\u0644\u0639\u0644\u0645"):
            print('chunk', row.get('chunk_id'), row.get('entity_type'), row.get('entity_id'), row.get('source_anchor'))
            print('head', txt[:140])
            break
