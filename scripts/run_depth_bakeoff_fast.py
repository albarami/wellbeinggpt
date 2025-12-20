"""
Fast Depth Bakeoff - 30 questions per model for quick comparison.
"""
import os
import sys
import time
import json
import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Load .env
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v:
            os.environ[k] = v

from openai import AzureOpenAI

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip()
DEPLOYMENTS = ["gpt-5-chat", "gpt-5.1", "gpt-5.2"]
SEED, TEMPERATURE, MAX_TOKENS = 1337, 0.1, 2000

CONCEPTS = [
    "الركيزة", "الركائز", "الحياة الطيبة", "الروحية", "الاجتماعية",
    "العقلية", "الفكرية", "الجسدية", "البدنية", "العاطفية", "النفسية",
    "الإيمان", "التوكل", "الصبر", "الرضا", "الشكر", "الإحسان",
    "العدل", "الحكمة", "التوازن", "العبادة", "الأخلاق", "الاستقامة",
    "التقوى", "التزكية", "الالتزام", "الذكر", "التفكر", "الخشوع"
]

PILLARS = {
    "spiritual": ["روحي", "روحية", "إيمان", "عبادة", "تقوى", "ذكر", "توكل"],
    "emotional": ["عاطفي", "عاطفية", "نفسي", "نفسية", "سكينة", "طمأنينة"],
    "intellectual": ["فكري", "فكرية", "عقلي", "عقلية", "تفكر", "حكمة"],
    "physical": ["جسدي", "جسدية", "بدني", "بدنية", "صحة"],
    "social": ["اجتماعي", "اجتماعية", "علاقات", "أسرة", "مجتمع"],
}

QUESTIONS = [
    # 10 synthesis
    {"id": "s01", "q": "كيف يؤدي الإطار إلى ازدهار الإنسان؟", "cat": "synthesis"},
    {"id": "s02", "q": "ما العلاقة بين جميع الركائز الخمس؟", "cat": "synthesis"},
    {"id": "s03", "q": "كيف تعمل الركائز معًا لتحقيق التوازن؟", "cat": "synthesis"},
    {"id": "s04", "q": "ما خطوات تحقيق الحياة الطيبة وفق الإطار؟", "cat": "synthesis"},
    {"id": "s05", "q": "ما المخاطر من إهمال إحدى الركائز؟", "cat": "synthesis"},
    {"id": "s06", "q": "ما الآليات الداخلية التي تربط الركائز ببعضها؟", "cat": "synthesis"},
    {"id": "s07", "q": "كيف يبني الإطار نموذجاً متكاملاً للإنسان؟", "cat": "synthesis"},
    {"id": "s08", "q": "ما الحلقات السببية بين الركائز الخمس؟", "cat": "synthesis"},
    {"id": "s09", "q": "كيف تتفاعل الحلقات السببية في نظام الركائز؟", "cat": "synthesis"},
    {"id": "s10", "q": "ما دور كل ركيزة في تحقيق الازدهار الشامل؟", "cat": "synthesis"},
    # 10 cross-pillar
    {"id": "c01", "q": "كيف ترتبط الحياة الروحية بالحياة الاجتماعية؟", "cat": "cross"},
    {"id": "c02", "q": "ما علاقة الإحسان بالتوكل؟", "cat": "cross"},
    {"id": "c03", "q": "كيف يؤثر التوازن الروحي على الصحة البدنية؟", "cat": "cross"},
    {"id": "c04", "q": "ما تأثير العلاقات الاجتماعية على الصحة النفسية؟", "cat": "cross"},
    {"id": "c05", "q": "كيف يعزز الإيمان الصحة العاطفية؟", "cat": "cross"},
    {"id": "c06", "q": "اربط بين الاستقامة والصحة الفكرية", "cat": "cross"},
    {"id": "c07", "q": "ما العلاقة بين الحكمة والعلاقات الاجتماعية؟", "cat": "cross"},
    {"id": "c08", "q": "كيف يرتبط الصبر بالصحة الجسدية؟", "cat": "cross"},
    {"id": "c09", "q": "ما العلاقة بين الشكر والسعادة النفسية؟", "cat": "cross"},
    {"id": "c10", "q": "كيف تؤثر الطمأنينة النفسية على الصحة الجسدية؟", "cat": "cross"},
    # 5 boundaries
    {"id": "b01", "q": "ما الفرق بين الصبر والرضا؟", "cat": "boundaries"},
    {"id": "b02", "q": "ما الفرق بين الإيمان والتوكل من حيث الأثر العملي؟", "cat": "boundaries"},
    {"id": "b03", "q": "ما حدود التوكل مع الأخذ بالأسباب؟", "cat": "boundaries"},
    {"id": "b04", "q": "فرّق بين العبادة والإيمان داخل الإطار", "cat": "boundaries"},
    {"id": "b05", "q": "ما التناقضات المحتملة بين الركائز وكيف تُحل؟", "cat": "boundaries"},
    # 5 natural chat
    {"id": "n01", "q": "أشعر بالضيق هذه الأيام، ماذا يقول الإطار عن ذلك؟", "cat": "chat"},
    {"id": "n02", "q": "كيف أتعامل مع القلق من وجهة نظر الإطار؟", "cat": "chat"},
    {"id": "n03", "q": "كيف أطبق الصبر عند الغضب؟", "cat": "chat"},
    {"id": "n04", "q": "كيف أقوي إيماني؟", "cat": "chat"},
    {"id": "n05", "q": "كيف أوازن بين عملي وحياتي الروحية؟", "cat": "chat"},
]

SYSTEM_PROMPT = """أنت عالم متخصص في إطار الحياة الطيبة (الركائز الخمس).
- أجب بالعربية الفصحى بأسلوب علمي عميق
- استخدم التنسيق المناسب (عناوين، نقاط، قوائم)
- اربط المفاهيم عبر الركائز المختلفة
- قدم تحليلاً معمقًا واختم بخلاصة تنفيذية
- إذا كان السؤال خارج نطاق الإطار، قل "خارج نطاق الإطار"
- لا تختلق معلومات"""


@dataclass 
class ModelStats:
    deployment: str
    results: list = field(default_factory=list)
    
    def avg_words(self): return sum(r["words"] for r in self.results) / max(1, len(self.results))
    def avg_concepts(self): return sum(r["concepts"] for r in self.results) / max(1, len(self.results))
    def avg_pillars(self): return sum(r["pillars"] for r in self.results) / max(1, len(self.results))
    def structure_rate(self): return sum(1 for r in self.results if r["structure"]) / max(1, len(self.results)) * 100
    def summary_rate(self): return sum(1 for r in self.results if r["summary"]) / max(1, len(self.results)) * 100
    
    def depth_score(self):
        return min(self.avg_words()/400, 1)*25 + min(self.avg_concepts()/8, 1)*25 + (self.structure_rate()/100)*25 + (self.summary_rate()/100)*25
    
    def cross_score(self):
        return min(self.avg_pillars()/4, 1)*50 + (sum(1 for r in self.results if r["pillars"]>=3)/max(1,len(self.results)))*50
    
    def composite(self):
        return self.depth_score()*0.45 + self.cross_score()*0.35 + 50*0.15 + 100*0.05


def count_pillars(text):
    found = set()
    for p, kws in PILLARS.items():
        if any(k in text for k in kws):
            found.add(p)
    return len(found)


def run():
    print("=" * 70)
    print("FAST DEPTH BAKEOFF (30 questions)")
    print("=" * 70)
    
    client = AzureOpenAI(azure_endpoint=ENDPOINT, api_key=API_KEY, api_version=API_VERSION)
    all_stats = []
    
    for dep in DEPLOYMENTS:
        print(f"\n{'-'*70}\nTESTING: {dep}\n{'-'*70}")
        stats = ModelStats(deployment=dep)
        
        for i, q in enumerate(QUESTIONS):
            print(f"  [{i+1}/30] {q['id']}: {q['q'][:35]}... ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model=dep, messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q["q"]}],
                    max_completion_tokens=MAX_TOKENS, temperature=TEMPERATURE, seed=SEED, timeout=90
                )
                ms = int((time.perf_counter()-t0)*1000)
                # Robust content extraction (gpt-5.1 returns list format)
                raw = resp.choices[0].message.content
                if raw is None:
                    content = ""
                elif isinstance(raw, str):
                    content = raw.strip()
                elif isinstance(raw, list):
                    parts = []
                    for item in raw:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            parts.append(item.get("text", "") or item.get("content", ""))
                    content = "".join(parts).strip()
                else:
                    content = str(raw).strip()
                words = len(content.split())
                concepts = sum(1 for c in CONCEPTS if c in content)
                pillars = count_pillars(content)
                structure = bool(re.search(r'^\s*[-*]\s', content, re.MULTILINE) or re.search(r'^\s*\d+[.)]\s', content, re.MULTILINE))
                summary = any(m in content for m in ["خلاصة", "ملخص", "في الختام"])
                stats.results.append({"words": words, "concepts": concepts, "pillars": pillars, "structure": structure, "summary": summary})
                print(f"OK ({ms}ms) w={words} c={concepts} p={pillars}")
            except Exception as e:
                stats.results.append({"words": 0, "concepts": 0, "pillars": 0, "structure": False, "summary": False})
                print(f"ERROR: {e}")
        all_stats.append(stats)
    
    print("\n" + "="*70 + "\nRESULTS\n" + "="*70)
    ranked = sorted(all_stats, key=lambda s: -s.composite())
    print(f"\n{'Model':<15} {'Composite':>10} {'Depth':>8} {'Cross':>8} {'Words':>8} {'Concepts':>9} {'Pillars':>8}")
    print("-"*80)
    for s in ranked:
        print(f"{s.deployment:<15} {s.composite():>10.1f} {s.depth_score():>8.1f} {s.cross_score():>8.1f} {s.avg_words():>8.0f} {s.avg_concepts():>9.1f} {s.avg_pillars():>8.2f}")
    
    winner = ranked[0]
    print(f"\n*** WINNER: {winner.deployment} (composite={winner.composite():.1f}) ***")
    
    # Write report
    Path("eval/reports").mkdir(parents=True, exist_ok=True)
    with open("eval/reports/model_bakeoff_depth.md", "w", encoding="utf-8") as f:
        f.write(f"# Depth Bakeoff Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"## Results\n\n")
        f.write("| Model | Composite | Depth | Cross-Pillar | Avg Words | Avg Concepts | Avg Pillars |\n")
        f.write("|-------|-----------|-------|--------------|-----------|--------------|-------------|\n")
        for s in ranked:
            f.write(f"| {s.deployment} | {s.composite():.1f} | {s.depth_score():.1f} | {s.cross_score():.1f} | {s.avg_words():.0f} | {s.avg_concepts():.1f} | {s.avg_pillars():.2f} |\n")
        f.write(f"\n## Winner\n\n**{winner.deployment}** with composite score {winner.composite():.1f}\n")
    print(f"\nReport: eval/reports/model_bakeoff_depth.md")


if __name__ == "__main__":
    run()
