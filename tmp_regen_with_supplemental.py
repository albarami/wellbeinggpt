import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Force OCR path (DOCX embedded image) + supplemental OCR auto-append
os.environ['INGEST_OCR_FROM_IMAGES']='required'

from apps.api.ingest.pipeline_framework import ingest_framework_docx

out_dir=Path('data/derived')
out_dir.mkdir(parents=True, exist_ok=True)
canonical_out=out_dir/'framework_2025-10_v1.ocr.canonical.json'
chunks_out=out_dir/'framework_2025-10_v1.ocr.chunks.jsonl'

summary=ingest_framework_docx(Path('docs/source/framework_2025-10_v1.docx'), canonical_out, chunks_out)
print('ingest_summary', summary)

# Quick sanity: confirm new terms exist somewhere in canonical JSON
canon=json.loads(canonical_out.read_text(encoding='utf-8'))
blob=json.dumps(canon, ensure_ascii=False)
for term in ['Ø§Ù„ØªØ¬Ø±Ø¯/Ø§Ù„Ø§Ø³ØªÙ‚Ù„Ø§Ù„ÙŠØ©','Ø§Ù„Ø§Ø®ØªØ¨Ø§Ø±/Ø§Ù„Ø§Ù…ØªØ­Ø§Ù†/Ø§Ù„Ø§Ø¨ØªÙ„Ø§Ø¡','Ø§Ù„Ø­ÙƒÙ…Ø©','ÙŠØ¤ØªÙŠ Ø§Ù„Ø­ÙƒÙ…Ø©','ÙƒÙ„ÙƒÙ… Ø±Ø§Ø¹']:
    print(term, term in blob)
