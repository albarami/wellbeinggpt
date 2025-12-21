"""
Generate an example edge trace row for validation.

This script simulates what a real trace log would look like.
"""

import json
from datetime import datetime
import uuid

# Example trace record (what the logger produces)
example_trace = {
    "trace_id": str(uuid.uuid4()),
    "timestamp": datetime.utcnow().isoformat(),
    "request_id": "req_abc123",
    "question": "ما العلاقة بين التوازن الجسدي والتوازن النفسي؟",
    "intent": "cross_pillar_path",
    "mode": "deep",
    "contract_outcome": "PASS_FULL",
    "candidate_count": 15,
    "candidate_count_logged": 15,
    "was_capped": False,
    "selected_count": 3,
    "candidate_edges": [
        {
            "edge_id": "edge_001",
            "from_type": "core_value",
            "from_id": "CV001",
            "to_type": "core_value", 
            "to_id": "CV002",
            "relation_type": "ENABLES",
            "span_count": 2,
            "quality_score": 0.75,
            "baseline_rank_position": 0,  # Ranked #1 by baseline
            "final_selected_rank": 0,     # Selected first
            "is_selected": True,  # POSITIVE (selected)
        },
        {
            "edge_id": "edge_002",
            "from_type": "core_value",
            "from_id": "CV001",
            "to_type": "pillar",
            "to_id": "P002",
            "relation_type": "REINFORCES",
            "span_count": 1,
            "quality_score": 0.55,
            "baseline_rank_position": 3,  # Ranked #4 by baseline
            "final_selected_rank": 1,     # Selected second
            "is_selected": True,  # POSITIVE (selected)
        },
        {
            "edge_id": "edge_003",
            "from_type": "pillar",
            "from_id": "P001",
            "to_type": "pillar",
            "to_id": "P002",
            "relation_type": "COMPLEMENTS",
            "span_count": 3,
            "quality_score": 0.60,
            "baseline_rank_position": 1,  # Ranked #2 by baseline
            "final_selected_rank": 2,     # Selected third
            "is_selected": True,  # POSITIVE (selected)
        },
        {
            "edge_id": "edge_004",
            "from_type": "pillar",
            "from_id": "P001",
            "to_type": "sub_value",
            "to_id": "SV005",
            "relation_type": "CONTAINS",
            "span_count": 1,
            "quality_score": 0.40,
            "baseline_rank_position": 4,  # Ranked #5 by baseline
            "final_selected_rank": -1,    # Not selected
            "is_selected": False,  # NEGATIVE (rejected)
        },
        {
            "edge_id": "edge_005",
            "from_type": "sub_value",
            "from_id": "SV001",
            "to_type": "sub_value",
            "to_id": "SV002",
            "relation_type": "RELATED_TO",
            "span_count": 0,
            "quality_score": 0.20,
            "baseline_rank_position": 5,  # Ranked last by baseline
            "final_selected_rank": -1,    # Not selected
            "is_selected": False,  # NEGATIVE (rejected - no evidence)
        },
        {
            "edge_id": "edge_006",
            "from_type": "core_value",
            "from_id": "CV003",
            "to_type": "core_value",
            "to_id": "CV004",
            "relation_type": "ENABLES",
            "span_count": 1,
            "quality_score": 0.50,
            "baseline_rank_position": 2,  # Ranked #3 by baseline
            "final_selected_rank": -1,    # Not selected
            "is_selected": False,  # NEGATIVE (rejected - lower quality than selected)
        },
    ],
    "selected_edge_ids": ["edge_001", "edge_002", "edge_003"],
    "rejected_edge_ids": ["edge_004", "edge_005", "edge_006"],
}

print("=" * 70)
print("EXAMPLE EDGE TRACE RECORD")
print("=" * 70)
print()
print("This is what a real training trace looks like:")
print()
print(json.dumps(example_trace, indent=2, ensure_ascii=False))
print()
print("=" * 70)
print("TRAINING SIGNAL SUMMARY")
print("=" * 70)
print()
print(f"Question: {example_trace['question']}")
print(f"Intent: {example_trace['intent']}")
print(f"Contract: {example_trace['contract_outcome']}")
print()
print(f"Total candidates: {example_trace['candidate_count']}")
print(f"Selected (positives): {example_trace['selected_count']}")
print(f"Rejected (negatives): {len(example_trace['rejected_edge_ids'])}")
print()
print("For ranking model training:")
print("  - Positive examples: edges with is_selected=True")
print("  - Negative examples: edges with is_selected=False")
print("  - Features: from_type, to_type, relation_type, span_count, quality_score")
print()
print("Pairwise ranking: selected edge should score higher than rejected edges")
print()
print("New fields for analysis:")
print("  - baseline_rank_position: where baseline ranked this edge (0=best)")
print("  - final_selected_rank: position in final selection (-1=not selected)")
print()
print("Edge edge_006 is interesting:")
print("  - Baseline ranked it #3 (quality_score=0.50)")
print("  - But it was NOT selected (final_selected_rank=-1)")
print("  - This is a HARD NEGATIVE: high-quality by baseline, but rejected")
print("  - These examples teach the model to discriminate beyond baseline")
print()
