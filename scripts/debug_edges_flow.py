#!/usr/bin/env python
"""Debug script to trace used_edges flow through the pipeline."""

import sys
import io
import requests
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    # Network question that previously failed
    question = "اختر قيمة محورية واحدة داخل الإطار ثم ابنِ شبكة تربطها بثلاث ركائز أخرى"
    
    print("=== DEBUG EDGES FLOW ===\n")
    print(f"Question: {question}\n")
    
    resp = requests.post('http://127.0.0.1:8002/ask/ui', json={
        'question': question,
        'language': 'ar',
        'mode': 'natural_chat',
        'engine': 'muhasibi'
    }, timeout=180)
    
    d = resp.json()
    
    # Check graph trace
    graph_trace = d.get('graph_trace', {})
    used_edges = graph_trace.get('used_edges', [])
    argument_chains = graph_trace.get('argument_chains', [])
    
    print(f"used_edges count: {len(used_edges)}")
    print(f"argument_chains count: {len(argument_chains)}")
    
    if used_edges:
        print("\nEdges found:")
        for i, e in enumerate(used_edges[:5]):
            print(f"  {i+1}. {e.get('from_node')} -> {e.get('to_node')} ({e.get('relation_type')})")
    else:
        print("\n*** NO USED_EDGES RETURNED ***")
        print("   This means ScholarReasoner didn't return edges")
    
    # Check answer for relation labels
    answer = d.get('final', {}).get('answer_ar', '')
    relation_labels = ["تمكين", "تعزيز", "تكامل", "إعانة", "شرط"]
    found_labels = [l for l in relation_labels if l in answer]
    
    print(f"\nRelation labels in answer: {found_labels if found_labels else 'None'}")
    
    if found_labels and not used_edges:
        print("\n*** BUG CONFIRMED ***")
        print("   Answer contains relation labels but no used_edges!")
        print("   LLM is hallucinating cross-pillar relationships")
    
    # Check contract
    contract = d.get('contract_outcome', '')
    reasons = d.get('contract_reasons', [])
    print(f"\nContract: {contract}")
    if reasons:
        print(f"Reasons: {reasons}")
    
    # Check trace for deep_mode
    for t in d.get('muhasibi_trace', []):
        if t.get('state') == 'LISTEN':
            print(f"\nLISTEN - entities: {t.get('detected_entities_count')}, keywords: {t.get('keywords_count')}")
    
    # Check if disclaimer was added
    if "لا توجد في الأدلة الحالية روابط" in answer or "لا توجد روابط" in answer:
        print("\n✓ Disclaimer about no grounded links was added to answer")
    
    # Show first 500 chars of answer
    print(f"\nAnswer preview (first 500 chars):\n{answer[:500]}...")

if __name__ == "__main__":
    main()
