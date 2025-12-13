import json
from pathlib import Path

doc_hash='57b1bc3b0bdb89d8266fe831c9a3e7cab2958861ae8a3590a68d96a56feb90c4'
base = Path('data/derived/supplemental_ocr')/doc_hash

# Ensure the evidence continuation screenshots attach to sub-value 'Ø³Ø¹Ø© Ø§Ù„Ø£ÙÙ‚'
patch = {
  '9d955a9fcd19bc14_hikmah_subvalue_saat_ufuq_evidence_D.json': 'Ø³Ø¹Ø© Ø§Ù„Ø£ÙÙ‚',
  'a3585e57a847fef4_hikmah_subvalue_saat_ufuq_more_E.json': 'Ø³Ø¹Ø© Ø§Ù„Ø£ÙÙ‚',
}

for fn, sub in patch.items():
    p = base/fn
    payload = json.loads(p.read_text(encoding='utf-8'))
    ctx = payload.get('context') or {}
    ctx['sub_value_name_ar'] = sub
    payload['context'] = ctx
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('patched', p)
