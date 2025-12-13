import json
from pathlib import Path

canonical_out=Path('data/derived/framework_2025-10_v1.suppfix.canonical.json')
canon=json.loads(canonical_out.read_text(encoding='utf-8'))

names=set()
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        for sv in cv.get('sub_values', []):
            names.add(sv.get('name_ar',''))

# Targets written as \u escapes (ASCII-only)
t1 = "\u0627\u0644\u062a\u062c\u0631\u062f/\u0627\u0644\u0627\u0633\u062a\u0642\u0644\u0627\u0644\u064a\u0629"
t2 = "\u0627\u0644\u0627\u062e\u062a\u0628\u0627\u0631/\u0627\u0644\u0627\u0645\u062a\u062d\u0627\u0646/\u0627\u0644\u0627\u0628\u062a\u0644\u0627\u0621"

print('t1_in', t1 in names)
print('t2_in', t2 in names)

# Show any sub-value names that include the same slash pattern (for debugging)
slashy=[n for n in names if '/' in n]
print('slashy_count', len(slashy))
print('slashy_sample', slashy[:20])
