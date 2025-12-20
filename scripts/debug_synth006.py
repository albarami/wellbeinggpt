"""Debug synth-006 flaky failure."""

import json
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    """Debug synth-006."""
    from apps.api.retrieve.seed_cache import SeedCache
    
    # Load the question
    dataset_path = "eval/datasets/regression_unexpected_fails.jsonl"
    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    
    synth006 = None
    for q in questions:
        if q.get("id") == "synth-006":
            synth006 = q
            break
    
    if not synth006:
        print("synth-006 not found")
        return
    
    print(f"Question ID: {synth006.get('id')}")
    print(f"Type: {synth006.get('type')}")
    
    # Check seed cache state
    cache = SeedCache.get_instance()
    print(f"\nSeed cache initialized: {cache.is_initialized()}")
    bundle = cache.get_global_bundle()
    if bundle:
        print(f"Pillar definitions in cache: {len(bundle.pillar_definitions)}")
        print(f"Cross-pillar edges in cache: {len(bundle.cross_pillar_edges)}")
        print(f"Total packets: {len(bundle.all_packets)}")
    else:
        print("No bundle cached yet")
    
    # Try to load the bundle from DB
    print("\n--- Attempting to load seed bundle from database ---")
    
    from apps.api.core.database import get_session
    from apps.api.retrieve.seed_cache import load_global_seed_bundle
    
    async with get_session() as session:
        try:
            bundle = await load_global_seed_bundle(session)
            print(f"Loaded pillar definitions: {len(bundle.pillar_definitions)}")
            print(f"Loaded cross-pillar edges: {len(bundle.cross_pillar_edges)}")
            print(f"Bundle is_empty: {bundle.is_empty}")
            
            if bundle.pillar_definitions:
                print("\nPillar definitions sample:")
                for p in bundle.pillar_definitions[:2]:
                    print(f"  - {p.get('chunk_id')}: {p.get('entity_name_ar', 'N/A')[:50]}")
            
            if bundle.cross_pillar_edges:
                print("\nCross-pillar edges sample:")
                for e in bundle.cross_pillar_edges[:2]:
                    print(f"  - {e.get('chunk_id')}: {e.get('text_ar', 'N/A')[:50]}")
        except Exception as ex:
            print(f"Error loading bundle: {ex}")
    
    print("\n--- Done ---")


if __name__ == "__main__":
    asyncio.run(main())
