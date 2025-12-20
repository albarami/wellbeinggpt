"""Quick test of regression questions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# Test bound-009 (system_limits_policy)
print("Testing bound-009: system limits policy question...")
response = requests.post(
    f"{BASE_URL}/ask/ui",
    json={
        "question": "ما حدود الربط بين الركائز غير المنصوص عليها؟",
        "model_deployment": "gpt-5.1",
    },
    timeout=120,
)
if response.status_code == 200:
    data = response.json()
    print(f"  contract_outcome: {data.get('contract_outcome')}")
    print(f"  abstained: {data.get('abstained')}")
    print(f"  citations_count: {len(data.get('citations_spans') or [])}")
    print(f"  answer_length: {len(data.get('answer_ar') or '')}")
    
    Path("d:/wellbeingqa/scripts/debug_bound009.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  (full response written to scripts/debug_bound009.json)")
else:
    print(f"  ERROR: {response.status_code}")

print()

# Test chat-011 (guidance_framework_chat) - requires natural_chat mode
print("Testing chat-011: guidance framework chat question (natural_chat mode)...")
response = requests.post(
    f"{BASE_URL}/ask/ui",
    json={
        "question": "أشعر بفقدان المعنى في حياتي، ماذا أفعل؟",
        "model_deployment": "gpt-5.1",
        "mode": "natural_chat",  # Important: must be natural_chat for guidance detection
    },
    timeout=120,
)
if response.status_code == 200:
    data = response.json()
    print(f"  contract_outcome: {data.get('contract_outcome')}")
    print(f"  abstained: {data.get('abstained')}")
    print(f"  citations_count: {len(data.get('citations_spans') or [])}")
    print(f"  answer_length: {len(data.get('answer_ar') or '')}")
    
    Path("d:/wellbeingqa/scripts/debug_chat011.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  (full response written to scripts/debug_chat011.json)")
else:
    print(f"  ERROR: {response.status_code}")

print()

# Test chat-019 (guidance_framework_chat) - requires natural_chat mode
print("Testing chat-019: self-improvement question (natural_chat mode)...")
response = requests.post(
    f"{BASE_URL}/ask/ui",
    json={
        "question": "أريد أن أكون شخصًا أفضل، من أين أبدأ؟",
        "model_deployment": "gpt-5.1",
        "mode": "natural_chat",  # Important: must be natural_chat for guidance detection
    },
    timeout=120,
)
if response.status_code == 200:
    data = response.json()
    print(f"  contract_outcome: {data.get('contract_outcome')}")
    print(f"  abstained: {data.get('abstained')}")
    print(f"  citations_count: {len(data.get('citations_spans') or [])}")
    print(f"  answer_length: {len(data.get('answer_ar') or '')}")
    
    Path("d:/wellbeingqa/scripts/debug_chat019.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  (full response written to scripts/debug_chat019.json)")
else:
    print(f"  ERROR: {response.status_code}")

print()
print("Done!")

