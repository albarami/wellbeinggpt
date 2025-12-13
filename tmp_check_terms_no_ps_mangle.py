import json
from pathlib import Path

data=json.loads(Path('data/derived/framework_2025-10_v1.ocr.canonical.json').read_text(encoding='utf-8'))

def find_snippet(term):
    import re
    s=json.dumps(data, ensure_ascii=False)
    m=re.search(re.escape(term), s)
    return bool(m)

terms=['Ø§Ù„ØªØ¬Ø±Ø¯/Ø§Ù„Ø§Ø³ØªÙ‚Ù„Ø§Ù„ÙŠØ©','Ø§Ù„Ø§Ø®ØªØ¨Ø§Ø±/Ø§Ù„Ø§Ù…ØªØ­Ø§Ù†/Ø§Ù„Ø§Ø¨ØªÙ„Ø§Ø¡','ÙƒÙ„ÙƒÙ… Ø±Ø§Ø¹','ÙŠØ¤ØªÙŠ Ø§Ù„Ø­ÙƒÙ…Ø©']
print({t: find_snippet(t) for t in terms})
