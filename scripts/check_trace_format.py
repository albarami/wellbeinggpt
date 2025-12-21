"""Check trace file format."""
import json

files_to_check = [
    "eval/output/bakeoff/gpt-5_1.jsonl",
    "eval/output/selective_bakeoff/bakeoff_selective_20251221_015000.jsonl",
    "eval/output/wellbeing__v1__f14d813f4519__seed1337__pv1__FULL_SYSTEM.jsonl",
]

for fpath in files_to_check:
    print(f"\n=== {fpath} ===")
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 1:
                    break
                data = json.loads(line)
                print(f"Keys: {list(data.keys())}")
                contract = data.get("contract_outcome") or data.get("contract")
                print(f"Contract: {contract}")
                gt = data.get("graph_trace")
                if gt:
                    if isinstance(gt, dict):
                        print(f"Graph trace keys: {list(gt.keys())}")
                        ue = gt.get("used_edges", [])
                        print(f"Used edges count: {len(ue)}")
                        if ue:
                            print(f"Edge sample keys: {list(ue[0].keys())}")
                    else:
                        print(f"Graph trace is type: {type(gt)}")
                else:
                    print("No graph_trace")
    except Exception as e:
        print(f"Error: {e}")
