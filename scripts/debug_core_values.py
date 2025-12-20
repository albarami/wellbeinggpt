#!/usr/bin/env python3
"""Check core values in spiritual pillar."""

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
        # Get core values for spiritual pillar (P001)
        result = await session.execute(text("""
            SELECT cv.id, cv.name_ar, p.name_ar as pillar_name
            FROM core_value cv
            JOIN pillar p ON cv.pillar_id = p.id
            WHERE p.id = 'P001'
            ORDER BY cv.name_ar
        """))
        rows = result.fetchall()
        print(f"Core values in Spiritual Pillar (الحياة الروحية): {len(rows)}")
        for r in rows:
            print(f"  - {r.name_ar} (id={r.id})")
        
        # Check if there are text_blocks with definitions
        print(f"\nDefinitions for spiritual core values:")
        for r in rows[:3]:
            result = await session.execute(text("""
                SELECT tb.text_ar, tb.type_label
                FROM text_block tb
                WHERE tb.owner_type = 'core_value' AND tb.owner_id = :cv_id
                LIMIT 1
            """), {"cv_id": r.id})
            tb = result.fetchone()
            if tb:
                print(f"  {r.name_ar}: {tb.text_ar[:100]}...")
            else:
                print(f"  {r.name_ar}: NO DEFINITION")


if __name__ == "__main__":
    asyncio.run(main())




