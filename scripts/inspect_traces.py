"""Inspect trace details to understand edge usage."""
import json
from pathlib import Path

SEMANTIC_TYPES = {"COMPLEMENTS", "ENABLES", "REINFORCES", "CONDITIONAL_ON", 
                  "INHIBITS", "TENSION_WITH", "RESOLVES_WITH"}

for f in Path("data/phase2/edge_traces/train").glob("*.jsonl"):
    with open(f, "r", encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            if i >= 5:
                break
            trace = json.loads(line)
            cands = trace.get("candidate_edges", [])
            selected_ids = set(trace.get("selected_edge_ids", []))
            semantic = [c for c in cands if c.get("relation_type", "").upper() in SEMANTIC_TYPES]
            
            intent = trace.get("intent", "unknown")
            print(f"Trace {i+1}: intent={intent}")
            print(f"  Candidates: {len(cands)} total, {len(semantic)} semantic")
            print(f"  Selected IDs count: {len(selected_ids)}")
            
            # Check overlap between selected_ids and candidate edge_ids
            cand_ids = set(str(c.get("edge_id", "")) for c in cands)
            overlap = selected_ids & cand_ids
            print(f"  IDs matching candidates: {len(overlap)}/{len(selected_ids)}")
            
            # Check which edges have is_selected=True
            used = [c for c in cands if c.get("is_selected")]
            print(f"  Used edges (is_selected=True): {len(used)}")
            
            # Check what relation types are used
            used_types = [c.get("relation_type") for c in used]
            print(f"  Used relation types: {used_types}")
            
            # Check semantic used
            semantic_used = [c for c in used if c.get("relation_type", "").upper() in SEMANTIC_TYPES]
            print(f"  Semantic used: {len(semantic_used)}")
            
            if semantic:
                print(f"  First 2 semantic candidates:")
                for s in semantic[:2]:
                    rel = s.get("relation_type")
                    selected = s.get("is_selected")
                    eid = s.get("edge_id", "")[:8]
                    in_selected = eid in str(selected_ids)[:100]
                    print(f"    {rel}: is_selected={selected}, id={eid}...")
            print()
