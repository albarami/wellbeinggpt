"""CLI: ingest external corpus sources from a provenance manifest.

Usage:
  python -m scripts.ingest_external_sources --manifest data/external_sources/manifest.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from apps.api.core.database import get_session
from apps.api.ingest.external_sources_ingest import ingest_external_sources
from apps.api.ingest.external_sources_manifest import load_external_manifest
from eval.datasets.source_loader import load_dotenv_if_present


async def _run(*, manifest_path: Path, embed: bool) -> None:
    rows = load_external_manifest(manifest_path=manifest_path)
    if not rows:
        raise SystemExit(f"No manifest rows found at: {manifest_path}")
    async with get_session() as session:
        res = await ingest_external_sources(session=session, rows=rows, repo_root=Path("."), embed=embed)
        await session.commit()
    for r in res:
        print(f"{r.source_id}: chunks={r.chunks} embeddings={r.embeddings} source_doc_id={r.source_doc_id}")


def main() -> None:
    load_dotenv_if_present()
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/external_sources/manifest.jsonl")
    p.add_argument("--no-embed", action="store_true", help="Skip embedding step")
    args = p.parse_args()
    asyncio.run(_run(manifest_path=Path(args.manifest), embed=not bool(args.no_embed)))


if __name__ == "__main__":
    main()

