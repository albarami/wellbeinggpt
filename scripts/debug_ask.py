#!/usr/bin/env python3
"""Debug script to test /ask endpoint responses."""

import sys
import os
from pathlib import Path

# Fix Windows encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
from httpx import AsyncClient, ASGITransport
from apps.api.main import app


async def test_question(question: str):
    """Test a single question and print debug info."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ask",
            json={"question": question},
            timeout=120.0,
        )
        data = resp.json()
        
        print(f"Status: {resp.status_code}")
        print(f"not_found: {data.get('not_found')}")
        print(f"confidence: {data.get('confidence')}")
        print(f"citations: {len(data.get('citations', []))}")
        print(f"entities: {[e.get('name_ar') for e in data.get('entities', [])]}")
        print(f"answer_ar (first 300 chars):")
        print(data.get("answer_ar", "")[:300])
        
        return data


async def main():
    # Test the failing question
    await test_question("اذكر ركائز الحياة الطيبة الخمس.")
    

if __name__ == "__main__":
    asyncio.run(main())




