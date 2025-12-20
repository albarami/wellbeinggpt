#!/usr/bin/env python3
"""Debug full /ask endpoint for pillar question."""

import sys
import os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
env_file = project_root / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key:
                    os.environ[key] = value

import asyncio
from httpx import AsyncClient, ASGITransport
from apps.api.main import app


async def main():
    question = "اذكر ركائز الحياة الطيبة الخمس."
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ask",
            json={"question": question},
            timeout=120.0,
        )
        data = resp.json()
        
        print(f"=== /ask Response ===")
        print(f"Status: {resp.status_code}")
        print(f"not_found: {data.get('not_found')}")
        print(f"confidence: {data.get('confidence')}")
        print(f"citations count: {len(data.get('citations', []))}")
        print(f"entities count: {len(data.get('entities', []))}")
        print(f"answer_ar (first 500):\n{data.get('answer_ar', '')[:500]}")
        
        if data.get('citations'):
            print(f"\n=== Citations ===")
            for i, c in enumerate(data.get('citations', [])[:5]):
                print(f"  {i+1}. chunk_id={c.get('chunk_id')}, source_anchor={c.get('source_anchor')}")


if __name__ == "__main__":
    asyncio.run(main())




