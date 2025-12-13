import asyncio
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()

from apps.api.core.database import get_session
from apps.api.ingest.pipeline_framework import ingest_framework_docx_from_bytes

async def main():
    p = Path('docs/source/framework_2025-10_v1.docx')
    b = p.read_bytes()
    run_id = str(uuid4())
    async with get_session() as session:
        summary = await ingest_framework_docx_from_bytes(
            session=session,
            file_content=b,
            file_name=p.name,
            run_id=run_id,
            canonical_json_path=Path('data/derived/framework_2025-10_v1.ocr.canonical.json'),
            chunks_jsonl_path=Path('data/derived/framework_2025-10_v1.ocr.chunks.jsonl'),
            framework_version='2025-10',
        )
    print('DB_INGEST_OK', summary)

asyncio.run(main())
