#!/usr/bin/env python3
"""Debug LLM response for pillar question."""

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
import json
from sqlalchemy import text
from apps.api.retrieve.entity_resolver import EntityResolver
from apps.api.retrieve.hybrid_retriever import HybridRetriever, RetrievalInputs
from apps.api.core.database import get_session
from apps.api.llm.gpt5_client_azure import ProviderConfig, create_provider, LLMRequest


async def main():
    question = "اذكر ركائز الحياة الطيبة الخمس."
    
    async with get_session() as session:
        # 1. Load entity resolver
        resolver = EntityResolver()
        pillars = (await session.execute(text("SELECT id, name_ar FROM pillar"))).fetchall()
        core_values = (await session.execute(text("SELECT id, name_ar FROM core_value"))).fetchall()
        sub_values = (await session.execute(text("SELECT id, name_ar FROM sub_value"))).fetchall()
        
        resolver.load_entities(
            pillars=[{"id": r.id, "name_ar": r.name_ar} for r in pillars],
            core_values=[{"id": r.id, "name_ar": r.name_ar} for r in core_values],
            sub_values=[{"id": r.id, "name_ar": r.name_ar} for r in sub_values],
            aliases_path="data/static/aliases_ar.json",
        )
        
        resolved = resolver.resolve(question)
        
        # 2. Get retrieval
        retriever = HybridRetriever()
        retriever._session = session
        
        entity_dicts = [
            {
                "type": r.entity_type.value,
                "id": r.entity_id,
                "name_ar": r.name_ar,
                "confidence": r.confidence,
            }
            for r in resolved
        ]
        
        merge = await retriever.retrieve(
            session,
            RetrievalInputs(
                query=question,
                resolved_entities=entity_dicts,
            ),
        )
        
        print(f"=== Evidence Packets for LLM ===")
        print(f"Count: {len(merge.evidence_packets)}")
        
        # 3. Call LLM directly
        cfg = ProviderConfig.from_env()
        print(f"\n=== LLM Config ===")
        print(f"Endpoint: {cfg.endpoint}")
        print(f"Configured: {cfg.is_configured()}")
        
        if not cfg.is_configured():
            print("ERROR: LLM not configured!")
            return
            
        provider = create_provider(cfg)
        
        # Directly call the provider to see raw response
        from apps.api.llm.muhasibi_llm_client import _read_prompt, _json_schema_for_interpreter, _sanitize_for_json
        from pathlib import Path as P
        
        prompt_dir = P(__file__).parent.parent / "apps" / "api" / "llm" / "prompts"
        system_prompt = (prompt_dir / "interpreter.md").read_text(encoding="utf-8")
        
        # Sanitize data
        sanitized_packets = []
        for p in merge.evidence_packets[:5]:  # Limit to 5 for testing
            sanitized = {}
            for k, v in p.items():
                if hasattr(v, '__class__') and v.__class__.__name__ == 'UUID':
                    sanitized[k] = str(v)
                else:
                    sanitized[k] = v
            sanitized_packets.append(sanitized)
        
        user_payload = {
            "question": question,
            "evidence_packets": sanitized_packets,
            "detected_entities": entity_dicts,
            "mode": "answer",
        }
        
        print(f"\n=== User Payload (truncated) ===")
        print(json.dumps(user_payload, ensure_ascii=False, indent=2)[:1000])
        
        req = LLMRequest(
            system_prompt=system_prompt,
            user_message=json.dumps(user_payload, ensure_ascii=False),
            response_format=_json_schema_for_interpreter(),
            temperature=0.2,
            max_tokens=1200,
        )
        
        print(f"\n=== Calling LLM ===")
        resp = await provider.complete(req)
        
        print(f"\n=== Raw LLM Response ===")
        print(f"Error: {resp.error}")
        print(f"Content (first 1000): {resp.content[:1000] if resp.content else 'None'}")
        print(f"Parsed JSON type: {type(resp.parsed_json)}")
        if resp.parsed_json:
            print(f"Parsed JSON keys: {resp.parsed_json.keys() if isinstance(resp.parsed_json, dict) else 'N/A'}")


if __name__ == "__main__":
    asyncio.run(main())




