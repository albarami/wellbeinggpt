import json
from pathlib import Path
from apps.api.ingest.pipeline_framework import ingest_framework_docx

out_dir=Path('data/derived')
out_dir.mkdir(parents=True, exist_ok=True)
canonical_out=out_dir/'framework_2025-10_v1.suppfix.canonical.json'
chunks_out=out_dir/'framework_2025-10_v1.suppfix.chunks.jsonl'

summary=ingest_framework_docx(Path('docs/source/framework_2025-10_v1.docx'), canonical_out, chunks_out)
print('summary', summary)

canon=json.loads(canonical_out.read_text(encoding='utf-8'))
# Prove the key new sub-values exist in canonical
names=[]
for p in canon.get('pillars', []):
    for cv in p.get('core_values', []):
        for sv in cv.get('sub_values', []):
            names.append(sv.get('name_ar',''))

for target in ['Ø§Ù„ØªØ¬Ø±Ø¯/Ø§Ù„Ø§Ø³ØªÙ‚Ù„Ø§Ù„ÙŠØ©','Ø§Ù„Ø§Ø®ØªØ¨Ø§Ø±/Ø§Ù„Ø§Ù…ØªØ­Ø§Ù†/Ø§Ù„Ø§Ø¨ØªÙ„Ø§Ø¡']:
    print(target, target in names)
