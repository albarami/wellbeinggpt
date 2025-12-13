import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from apps.api.core.schema_bootstrap import bootstrap_db
from apps.api.core.database import get_session
from apps.api.ingest.loader import load_canonical_json_to_db

CANON=Path('data/derived/framework_2025-10_v1.suppfix7.canonical.json')
canonical=json.loads(CANON.read_text(encoding='utf-8'))

async def main():
    # Ensure schema exists
    await bootstrap_db('db/schema.sql')
    async with get_session() as session:
        summary = await load_canonical_json_to_db(session, canonical, 'framework_2025-10_v1.docx')
    print('load_summary', summary)

asyncio.run(main())
