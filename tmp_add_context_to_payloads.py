import json
from pathlib import Path

doc_hash='57b1bc3b0bdb89d8266fe831c9a3e7cab2958861ae8a3590a68d96a56feb90c4'
base = Path('data/derived/supplemental_ocr')/doc_hash

# Add context hints to the specific new payloads so they attach under (Ø§Ù„Ø­ÙŠØ§Ø© Ø§Ù„ÙÙƒØ±ÙŠØ© -> Ø§Ù„Ø­ÙƒÙ…Ø©).
# We only adjust our Option-B generated JSONs.

targets = [
    '55a32d33429b27df_hikmah_subvalue_knowledge_B.json',
    '091ee6ad24b358ce_hikmah_subvalue_saat_ufuq_concept_C.json',
    '9d955a9fcd19bc14_hikmah_subvalue_saat_ufuq_evidence_D.json',
    'a3585e57a847fef4_hikmah_subvalue_saat_ufuq_more_E.json',
]

for name in targets:
    p = base/name
    if not p.exists():
        print('missing', p)
        continue
    payload = json.loads(p.read_text(encoding='utf-8'))
    payload['context'] = {
        'pillar_name_ar': '\u0627\u0644\u062d\u064a\u0627\u0629 \u0627\u0644\u0641\u0643\u0631\u064a\u0629',
        'core_value_name_ar': '\u0627\u0644\u062d\u0643\u0645\u0629',
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('updated_context', p)
