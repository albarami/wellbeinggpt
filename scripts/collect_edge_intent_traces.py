"""
Collect traces specifically for edge-intent questions.

These questions are designed to trigger cross_pillar, network, tension intents
which REQUIRE edge usage in their answers.
"""

import json
import time
from pathlib import Path
import httpx

API_URL = "http://localhost:8002/ask/ui"
TIMEOUT = 120.0

# Questions that should trigger edge-intents (cross_pillar, network, tension)
EDGE_INTENT_QUESTIONS = [
    # Cross-pillar relationship questions
    "ما العلاقة بين البعد الروحي والبعد البدني؟",
    "كيف يتكامل البعد النفسي مع البعد الاجتماعي؟",
    "ما الصلة بين التوازن البدني والتوازن الروحي؟",
    "كيف يؤثر البعد الروحي على البعد النفسي؟",
    "ما العلاقة بين الإيمان والصحة النفسية؟",
    "كيف يرتبط البعد المادي بالبعد الروحي؟",
    "ما التفاعل بين الأبعاد الخمسة للإطار؟",
    "كيف تتكامل الركائز الخمس معًا؟",
    
    # Network/connection questions
    "اربط بين قيمة الشكر وبقية القيم المحورية",
    "ما الروابط بين الصبر والقيم الأخرى؟",
    "اشرح شبكة العلاقات بين قيم الإطار",
    "كيف ترتبط القيم المحورية ببعضها؟",
    "ما الروابط بين التوكل وباقي القيم؟",
    "اربط بين الإحسان والقيم الأخرى",
    
    # Tension/reconciliation questions
    "هل هناك توتر بين الزهد والسعي للرزق؟",
    "كيف نوفق بين الخوف والرجاء؟",
    "ما التوازن بين العزلة والاختلاط؟",
    "كيف نوفق بين التوكل والأخذ بالأسباب؟",
    
    # Cross-pillar path questions
    "ما المسار من البعد الروحي إلى البعد الاجتماعي؟",
    "كيف ينتقل الأثر من البعد النفسي إلى البعد البدني؟",
    "ما خطوات الانتقال من الإيمان إلى العمل؟",
    
    # Global synthesis questions
    "كيف تسهم الركائز الخمس في تحقيق الحياة الطيبة؟",
    "ما دور التكامل بين الأبعاد في تحقيق الازدهار؟",
    "كيف يحقق الإطار التوازن الشامل للإنسان؟",
    
    # Value-level cross-pillar
    "ما العلاقة بين الخشوع والسكينة؟",
    "كيف يرتبط الصدق بالطمأنينة؟",
    "ما الصلة بين التقوى والصحة النفسية؟",
    "كيف يؤثر الذكر على التوازن النفسي؟",
    "ما العلاقة بين الإخلاص والرضا؟",
]

def count_traces():
    """Count current training traces."""
    trace_dir = Path("data/phase2/edge_traces/train")
    if not trace_dir.exists():
        return 0
    count = 0
    for f in trace_dir.glob("*.jsonl"):
        with open(f, "r", encoding="utf-8") as fp:
            count += sum(1 for _ in fp)
    return count

def main():
    print("=" * 70)
    print("EDGE-INTENT TRACE COLLECTION")
    print("=" * 70)
    
    initial = count_traces()
    print(f"Initial traces: {initial}")
    print(f"Questions to send: {len(EDGE_INTENT_QUESTIONS)}")
    
    outcomes = {"PASS_FULL": 0, "PASS_PARTIAL": 0, "FAIL": 0, "OTHER": 0}
    
    with httpx.Client() as client:
        for i, question in enumerate(EDGE_INTENT_QUESTIONS, 1):
            try:
                response = client.post(
                    API_URL,
                    json={"question": question, "mode": "answer"},
                    timeout=TIMEOUT,
                )
                if response.status_code == 200:
                    data = response.json()
                    outcome = data.get("contract", {}).get("outcome", "UNKNOWN")
                    if outcome == "PASS_FULL":
                        outcomes["PASS_FULL"] += 1
                    elif outcome == "PASS_PARTIAL":
                        outcomes["PASS_PARTIAL"] += 1
                    elif outcome == "FAIL":
                        outcomes["FAIL"] += 1
                    else:
                        outcomes["OTHER"] += 1
                else:
                    outcomes["OTHER"] += 1
                    outcome = f"HTTP_{response.status_code}"
            except Exception as e:
                outcomes["OTHER"] += 1
                outcome = f"ERROR"
            
            current = count_traces()
            print(f"  [{i:2}/{len(EDGE_INTENT_QUESTIONS)}] {outcome} | traces={current}")
            time.sleep(0.1)
    
    final = count_traces()
    print()
    print("=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Total traces: {final}")
    print(f"New traces: {final - initial}")
    print(f"Outcomes: {outcomes}")

if __name__ == "__main__":
    main()
