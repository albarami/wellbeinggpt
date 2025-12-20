"""Test single /ask request."""
import requests
import time
import json

t0 = time.perf_counter()
try:
    print("Sending request to /ask...")
    r = requests.post(
        'http://localhost:8000/ask',
        json={
            'question': 'ما هو الإيمان؟',
            'model_deployment': 'gpt-5-chat'
        },
        timeout=300
    )
    elapsed = time.perf_counter() - t0
    print(f"Status: {r.status_code}")
    print(f"Time: {elapsed:.1f}s")
    if r.status_code == 200:
        data = r.json()
        print(f"Answer length: {len(data.get('answer_ar', ''))}")
        print(f"Citations: {len(data.get('citations', []))}")
        print(f"Contract: {data.get('contract_outcome', '')}")
    else:
        print(f"Response: {r.text[:500]}")
except Exception as e:
    elapsed = time.perf_counter() - t0
    print(f"Error after {elapsed:.1f}s: {e}")


