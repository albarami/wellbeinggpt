"""Simple API test."""
import requests
import json

payload = {
    "question": "ما المنظور الكلي للإطار؟",
    "mode": "answer",
    "lang": "ar"
}

response = requests.post("http://127.0.0.1:8000/ask/ui", json=payload, timeout=120)
print(f"Status: {response.status_code}")
if response.status_code != 200:
    print(f"Error: {response.text[:500]}")
else:
    data = response.json()
    print(f"Contract: {data.get('contract_outcome')}")
    print(f"Citations: {len(data.get('citations_spans', []))}")
