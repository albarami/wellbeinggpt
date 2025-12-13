import hashlib, json
from pathlib import Path

DOC_HASH='57b1bc3b0bdb89d8266fe831c9a3e7cab2958861ae8a3590a68d96a56feb90c4'
base=Path('data/derived/supplemental_ocr')/DOC_HASH

CARE_FILE=base/'dfe437d05e40c0cb_social_core_care_structure.json'
RESP_FILE=base/'42e842f0b4048002_social_core_responsibility_structure.json'
MAWADDA_FILE=base/'715f520ad4324f4a_social_care_mawadda_rahma_02.json'

care=json.loads(CARE_FILE.read_text(encoding='utf-8'))
care['lines']=[
  '1) \u0627\u0644\u062a\u0639\u0627\u0637\u0641',
  '2) \u0627\u0644\u0645\u0648\u062f\u0629 \u0648\u0627\u0644\u0631\u062d\u0645\u0629 (\u0642\u064a\u0645\u0629 \u0645\u062a\u0644\u0627\u0632\u0645\u0629)',
  '3) \u0627\u0644\u0627\u0647\u062a\u0645\u0627\u0645',
  '4) \u0627\u0644\u0643\u0631\u0645',
  '5) \u0627\u0644\u0648\u0641\u0627\u0621',
]
care_joined='\n'.join(care['lines']).strip()+'\n'
care['image_sha256']=hashlib.sha256(care_joined.encode('utf-8')).hexdigest()
CARE_FILE.write_text(json.dumps(care, ensure_ascii=False, indent=2), encoding='utf-8')

resp=json.loads(RESP_FILE.read_text(encoding='utf-8'))
resp['lines']=[
  '1) \u0627\u0644\u0639\u062f\u0627\u0644\u0629',
  '2) \u0627\u0644\u0627\u0633\u062a\u062e\u0644\u0627\u0641 \u0648\u0627\u0644\u0639\u0645\u0631\u0627\u0646',
  '3) \u0627\u0644\u0645\u062d\u0627\u0633\u0628\u0629 \u0648\u0627\u0644\u0645\u0633\u0627\u0621\u0644\u0629 (\u0642\u064a\u0645\u0629 \u0645\u062a\u0644\u0627\u0632\u0645\u0629)',
  '4) \u0627\u0644\u0627\u0644\u062a\u0632\u0627\u0645',
  '5) \u0627\u0644\u0641\u0627\u0639\u0644\u064a\u0629 (\u0627\u0644\u0623\u062f\u0627\u0621 \u0648\u0627\u0644\u0627\u0646\u062c\u0627\u0632)',
]
resp_joined='\n'.join(resp['lines']).strip()+'\n'
resp['image_sha256']=hashlib.sha256(resp_joined.encode('utf-8')).hexdigest()
RESP_FILE.write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding='utf-8')

mw=json.loads(MAWADDA_FILE.read_text(encoding='utf-8'))
mw.setdefault('context', {})['sub_value_name_ar']='\u0627\u0644\u0645\u0648\u062f\u0629 \u0648\u0627\u0644\u0631\u062d\u0645\u0629 (\u0642\u064a\u0645\u0629 \u0645\u062a\u0644\u0627\u0632\u0645\u0629)'
# keep mw['image_sha256'] as-is because lines unchanged
MAWADDA_FILE.write_text(json.dumps(mw, ensure_ascii=False, indent=2), encoding='utf-8')

print('care sha', care['image_sha256'][:16])
print('resp sha', resp['image_sha256'][:16])
print('mw sha', mw['image_sha256'][:16])
