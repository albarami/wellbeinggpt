"""Check edge format in FULL_SYSTEM files."""
import json
from pathlib import Path

trace_dir = Path("eval/output")

for jsonl_file in sorted(trace_dir.glob("*FULL_SYSTEM*.jsonl"))[:5]:
    print(f"\n=== {jsonl_file.name} ===")
    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                data = json.loads(line)
                gt = data.get("graph_trace", {})
                if isinstance(gt, dict):
                    print(f"  graph_trace keys: {list(gt.keys())}")
                    for key in gt.keys():
                        val = gt[key]
                        if isinstance(val, list) and val:
                            print(f"    {key}: {len(val)} items, first type: {type(val[0])}")
                            if isinstance(val[0], dict):
                                print(f"      keys: {list(val[0].keys())}")
                            elif isinstance(val[0], str):
                                print(f"      sample: {val[0][:100]}")
                            break
                else:
                    print(f"  graph_trace is {type(gt)}")
    except Exception as e:
        print(f"Error: {e}")
