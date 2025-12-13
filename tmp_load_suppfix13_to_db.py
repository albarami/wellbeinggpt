import asyncio, json
from pathlib import Path
from apps.api.core.database import get_session
from apps.api.ingest.loader import load_canonical_json_to_db

CANON=Path('data/derived/framework_2025-10_v1.suppfix13.canonical.json')
canonical=json.loads(CANON.read_text(encoding='utf-8'))

async def main():
    async with get_session() as session:
        summary = await load_canonical_json_to_db(session, canonical, 'framework_2025-10_v1.docx')
    print('load_summary', summary)

asyncio.run(main())
