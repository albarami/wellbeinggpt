#!/usr/bin/env python3
"""Check graph edges for pillar to core_value."""

import sys
import os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip()

import asyncio
from sqlalchemy import text
from apps.api.core.database import get_session


async def main():
    async with get_session() as session:
        # Check CONTAINS edges from P001 (spiritual pillar)
        print("=== CONTAINS edges from P001 (الحياة الروحية) ===")
        result = await session.execute(text("""
            SELECT from_type, from_id, to_type, to_id, rel_type
            FROM edge
            WHERE from_id = 'P001' OR to_id = 'P001'
            ORDER BY rel_type, to_type
        """))
        rows = result.fetchall()
        for r in rows:
            print(f"  {r.from_type}:{r.from_id} --{r.rel_type}--> {r.to_type}:{r.to_id}")
        
        print(f"\n=== Core values under P001 ===")
        result = await session.execute(text("""
            SELECT id, name_ar, pillar_id
            FROM core_value
            WHERE pillar_id = 'P001'
        """))
        rows = result.fetchall()
        for r in rows:
            print(f"  {r.name_ar} (id={r.id})")
        
        print(f"\n=== Chunks for core values under P001 ===")
        for r in rows:
            result = await session.execute(text("""
                SELECT chunk_id, chunk_type, text_ar
                FROM chunk
                WHERE entity_type = 'core_value' AND entity_id = :cv_id
                LIMIT 3
            """), {"cv_id": r.id})
            chunks = result.fetchall()
            print(f"\n  {r.name_ar} ({r.id}): {len(chunks)} chunks")
            for c in chunks:
                print(f"    - {c.chunk_id} ({c.chunk_type}): {c.text_ar[:80]}...")


if __name__ == "__main__":
    asyncio.run(main())




