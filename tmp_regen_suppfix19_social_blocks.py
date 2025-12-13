import json
from pathlib import Path
from apps.api.ingest.pipeline_framework import ingest_framework_docx

out_dir=Path('data/derived'); out_dir.mkdir(parents=True, exist_ok=True)
canonical_out=out_dir/'framework_2025-10_v1.suppfix19.canonical.json'
chunks_out=out_dir/'framework_2025-10_v1.suppfix19.chunks.jsonl'

summary=ingest_framework_docx(Path('docs/source/framework_2025-10_v1.docx'), canonical_out, chunks_out)
print('summary', summary)

canon=json.loads(canonical_out.read_text(encoding='utf-8'))
blocks=canon.get('supplemental_text_blocks') or []
print('supplemental_text_blocks', len(blocks))
print('first5', [ (b.get('entity_type'), b.get('entity_id'), b.get('block_type'), (b.get('source_anchor') or {}).get('source_anchor')) for b in blocks[:5] ])
