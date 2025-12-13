import hashlib, json
from pathlib import Path

DOC_HASH='57b1bc3b0bdb89d8266fe831c9a3e7cab2958861ae8a3590a68d96a56feb90c4'
base=Path('data/derived/supplemental_ocr')/DOC_HASH

files=[
  'dfe437d05e40c0cb_social_core_care_structure.json',
  '42e842f0b4048002_social_core_responsibility_structure.json',
  '715f520ad4324f4a_social_care_mawadda_rahma_02.json',
]

for fn in files:
    p=base/fn
    payload=json.loads(p.read_text(encoding='utf-8'))
    lines=payload.get('lines') or []

    if 'social_core_care_structure' in fn:
        lines=[
          '1) Ø§Ù„ØªØ¹Ø§Ø·Ù',
          '2) Ø§Ù„Ù…ÙˆØ¯Ø© ÙˆØ§Ù„Ø±Ø­Ù…Ø© (Ù‚ÙŠÙ…Ø© Ù…ØªÙ„Ø§Ø²Ù…Ø©)',
          '3) Ø§Ù„Ø§Ù‡ØªÙ…Ø§Ù…',
          '4) Ø§Ù„ÙƒØ±Ù…',
          '5) Ø§Ù„ÙˆÙØ§Ø¡',
        ]
        payload['lines']=lines

    if 'social_core_responsibility_structure' in fn:
        lines=[
          '1) Ø§Ù„Ø¹Ø¯Ø§Ù„Ø©',
          '2) Ø§Ù„Ø§Ø³ØªØ®Ù„Ø§Ù ÙˆØ§Ù„Ø¹Ù…Ø±Ø§Ù†',
          '3) Ø§Ù„Ù…Ø­Ø§Ø³Ø¨Ø© ÙˆØ§Ù„Ù…Ø³Ø§Ø¡Ù„Ø© (Ù‚ÙŠÙ…Ø© Ù…ØªÙ„Ø§Ø²Ù…Ø©)',
          '4) Ø§Ù„Ø§Ù„ØªØ²Ø§Ù…',
          '5) Ø§Ù„ÙØ§Ø¹Ù„ÙŠØ© (Ø§Ù„Ø£Ø¯Ø§Ø¡ ÙˆØ§Ù„Ø§Ù†Ø¬Ø§Ø²)',
        ]
        payload['lines']=lines

    if 'social_care_mawadda_rahma_02' in fn:
        payload.setdefault('context', {})['sub_value_name_ar']='Ø§Ù„Ù…ÙˆØ¯Ø© ÙˆØ§Ù„Ø±Ø­Ù…Ø© (Ù‚ÙŠÙ…Ø© Ù…ØªÙ„Ø§Ø²Ù…Ø©)'

    joined='\n'.join(payload.get('lines') or []).strip()+'\n'
    payload['image_sha256']=hashlib.sha256(joined.encode('utf-8')).hexdigest()

    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('updated', fn, '->', payload['image_sha256'][:16])

print('done')
