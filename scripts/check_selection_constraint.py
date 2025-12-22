"""
Check if used_edges are constrained to selected_edges in traces.

For training labels to be meaningful, used_edges should be a subset of
the selected top-K candidates (selected_edge_ids).
"""

import json
from pathlib import Path

TRACE_DIR = Path("data/phase2/edge_traces/train")

total_traces = 0
traces_with_constraint_met = 0
traces_with_constraint_violated = 0
traces_without_used_edges = 0

violations = []

for trace_file in sorted(TRACE_DIR.glob("*.jsonl")):
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                trace = json.loads(line)
                total_traces += 1
                
                selected_ids = set(trace.get("selected_edge_ids", []))
                
                # Find used edges (is_selected=True in candidates)
                candidates = trace.get("candidate_edges", [])
                used_ids = set(
                    str(c.get("edge_id", "")) 
                    for c in candidates 
                    if c.get("is_selected", False)
                )
                
                if not used_ids:
                    traces_without_used_edges += 1
                    continue
                
                # Check: used_ids should be subset of selected_ids
                if used_ids <= selected_ids:
                    traces_with_constraint_met += 1
                else:
                    traces_with_constraint_violated += 1
                    extra = used_ids - selected_ids
                    violations.append({
                        "question": trace.get("question", "")[:50],
                        "intent": trace.get("intent", ""),
                        "selected_count": len(selected_ids),
                        "used_count": len(used_ids),
                        "extra_used_ids": list(extra)[:3],  # First 3 violators
                    })
                    
            except json.JSONDecodeError:
                pass

print("=" * 70)
print("SELECTION CONSTRAINT CHECK: used_edges ⊆ selected_edges")
print("=" * 70)
print()
print(f"Total traces analyzed: {total_traces}")
print(f"Traces without used edges: {traces_without_used_edges}")
print(f"Traces with constraint MET: {traces_with_constraint_met}")
print(f"Traces with constraint VIOLATED: {traces_with_constraint_violated}")
print()

if traces_with_constraint_violated > 0:
    violation_rate = traces_with_constraint_violated / max(1, total_traces - traces_without_used_edges)
    print(f"Violation rate: {violation_rate:.1%}")
    print()
    print("Sample violations:")
    for v in violations[:5]:
        print(f"  Intent: {v['intent']}")
        print(f"  Question: {v['question']}...")
        print(f"  Selected: {v['selected_count']}, Used: {v['used_count']}")
        print(f"  Extra IDs (not in selected): {v['extra_used_ids']}")
        print()
else:
    print("✓ All used edges are within selected edges - constraint satisfied!")
