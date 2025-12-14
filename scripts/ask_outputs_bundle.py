"""
/ask Outputs Bundle

Runs a fixed set of questions through the real FastAPI app (no external server required),
and writes a bundle JSON with:
- 3 in-corpus questions
- 3 cross-pillar questions
- 3 out-of-scope refusal questions

Usage:
    python scripts/ask_outputs_bundle.py

Env vars required:
    DATABASE_URL
    (optional) AZURE_OPENAI_* for LLM-enabled answers; if missing, system should refuse safely.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from apps.api.main import app


def _require_env(name: str) -> str:
    val = (os.getenv(name) or "").strip()
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


ASK_CASES = [
    # In-corpus (structure)
    {"id": "in-01", "kind": "in_corpus", "engine": "muhasibi", "question": "اذكر ركائز الحياة الطيبة الخمس."},
    {"id": "in-02", "kind": "in_corpus", "engine": "muhasibi", "question": "ما القيم الكلية في الركيزة الروحية الطيبة؟"},
    {"id": "in-03", "kind": "in_corpus", "engine": "muhasibi", "question": "عرّف التوحيد واذكر التأصيل."},
    # Cross-pillar
    {"id": "x-01", "kind": "cross_pillar", "engine": "muhasibi", "question": "اربط بين التوكل والتوازن العاطفي: أين ورد التوكل؟ وأين ورد التوازن؟"},
    {"id": "x-02", "kind": "cross_pillar", "engine": "muhasibi", "question": "كيف يرتبط الصبر بالتوازن العاطفي وبالإنتاجية؟ أعطني شواهد من النص."},
    {"id": "x-03", "kind": "cross_pillar", "engine": "muhasibi", "question": "اربط بين التقويم الذاتي (فكري) والمحاسبة والمساءلة (اجتماعي)."},
    # Out-of-scope
    {"id": "oos-01", "kind": "out_of_scope", "engine": "muhasibi", "question": "ما حكم صيام يوم الجمعة منفردًا؟"},
    {"id": "oos-02", "kind": "out_of_scope", "engine": "muhasibi", "question": "اذكر فوائد طبية لدواء الباراسيتامول."},
    {"id": "oos-03", "kind": "out_of_scope", "engine": "muhasibi", "question": "من هو مؤلف كتاب كذا؟"},
]


async def _run() -> dict[str, Any]:
    out: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "vector_backend": os.getenv("VECTOR_BACKEND", "disabled").lower(),
        "llm_configured": bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY")),
        "cases": [],
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for c in ASK_CASES:
            payload = {"question": c["question"], "engine": c["engine"], "language": "ar", "mode": "answer"}
            r = await client.post("/ask", json=payload, timeout=120.0)
            data = r.json()
            out["cases"].append(
                {
                    "id": c["id"],
                    "kind": c["kind"],
                    "question": c["question"],
                    "engine": c["engine"],
                    "http_status": r.status_code,
                    "response": data,
                }
            )

    return out


def main() -> None:
    load_dotenv()
    _require_env("DATABASE_URL")
    bundle = asyncio.run(_run())
    path = Path("ask_outputs_bundle.json")
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {path}")


if __name__ == "__main__":
    main()

