#!/usr/bin/env python
"""Debug script to test Muhasibi trace output."""

import sys
import io
import requests
import json

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    resp = requests.post('http://127.0.0.1:8002/ask/ui', json={
        'question': 'ما هي الحياة الطيبة؟',
        'language': 'ar',
        'mode': 'natural_chat',
        'engine': 'muhasibi'
    }, timeout=90)
    
    d = resp.json()
    print('=== MUHASIBI TRACE ===\n')
    for t in d.get('muhasibi_trace', []):
        state = t.get('state', 'UNKNOWN')
        elapsed = t.get('elapsed_s', 0)
        print(f"--- {state} ({elapsed:.3f}s) ---")
        for k, v in t.items():
            if k not in ('state', 'elapsed_s', 'mode', 'language'):
                # Truncate long strings
                if isinstance(v, str) and len(v) > 100:
                    v = v[:100] + "..."
                print(f"  {k}: {v}")
        print()

if __name__ == "__main__":
    main()
