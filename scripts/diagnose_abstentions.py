"""Diagnose abstention reasons for the 3 questions failing on ALL models."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "eval/output/bakeoff_depth_v1_system"
TARGET_QIDS = ["bound-009", "chat-011", "chat-019"]
DEPLOYMENTS = ["gpt-5-chat", "gpt-5.1", "gpt-5.2"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    rows.append(json.loads(s))
                except Exception:
                    pass
    return rows


def main():
    # Write to file to avoid Windows encoding issues
    out_path = REPO / "eval/output/abstention_diagnosis.json"
    results = {}
    
    for qid in TARGET_QIDS:
        results[qid] = {}
        
        for dep in DEPLOYMENTS:
            path = OUT_DIR / f"{dep}.jsonl"
            rows = load_jsonl(path)
            
            # Find this question
            for r in rows:
                if r.get("id") == qid:
                    trace_raw = r.get("muhasibi_trace")
                    # Handle trace as list or dict
                    if isinstance(trace_raw, list):
                        trace = {"states": trace_raw}
                    elif isinstance(trace_raw, dict):
                        trace = trace_raw
                    else:
                        trace = {}
                    
                    results[qid][dep] = {
                        "contract_outcome": r.get("contract_outcome"),
                        "contract_reasons": r.get("contract_reasons"),
                        "abstained": r.get("abstained"),
                        "citations_count": len(r.get("citations_spans") or []),
                        "edges_count": len((r.get("graph_trace") or {}).get("used_edges") or []),
                        "muhasibi_trace": trace,
                        "answer_length": len(r.get("answer_ar") or ""),
                    }
                    break
    
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote diagnosis to: {out_path}")
    
    # Print summary (ASCII only)
    for qid, models in results.items():
        print(f"\n{qid}:")
        for dep, data in models.items():
            print(f"  [{dep}] outcome={data['contract_outcome']} intent={data.get('muhasibi_intent')} cites={data['citations_count']} edges={data['edges_count']}")


if __name__ == "__main__":
    main()

