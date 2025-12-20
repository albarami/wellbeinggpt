"""
Direct System Bakeoff - Test models with actual Muhasibi prompts
================================================================
Calls Azure OpenAI directly using the same prompts the system uses,
bypassing the slow HTTP/database layer but using identical LLM config.
"""

import os
import sys
import time
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

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

# Config
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip()
DEPLOYMENTS = ["gpt-5-chat", "gpt-5.1", "gpt-5.2"]

# Control variables (IDENTICAL for all models)
SEED = 1337
TEMPERATURE = 0.1
MAX_TOKENS = 2000

# This is the ACTUAL system prompt from the Muhasibi interpreter
MUHASIBI_SYSTEM_PROMPT = """أنت مُحاسِبي، عالم إسلامي متخصص في إطار "ركائز الحياة الطيبة" (الركائز الخمس: الروحية، العاطفية، الفكرية، البدنية، الاجتماعية).

### المبادئ الأساسية:
1. **الاستناد إلى الأدلة فقط**: كل ما تقوله يجب أن يستند إلى نصوص الإطار المقدمة
2. **الربط بين الركائز**: وضّح العلاقات والتأثيرات المتبادلة بين الركائز المختلفة
3. **الأمانة العلمية**: إذا لم يكن هناك دليل واضح، اذكر ذلك صراحة
4. **الوضوح والعمق**: قدم إجابات واضحة ومعمقة تناسب السائل

### قواعد الإجابة:
- استخدم العربية الفصحى
- نظّم الإجابة بعناوين ونقاط واضحة
- اربط المفاهيم عبر الركائز حيثما أمكن
- اختم بخلاصة تنفيذية أو توصيات عملية
- إذا كان السؤال خارج نطاق الإطار، قل "هذا الموضوع خارج نطاق إطار الحياة الطيبة"
- لا تختلق معلومات أو أدلة غير موجودة"""

# Framework concepts for detection
CONCEPTS = [
    "الركيزة", "الركائز", "الحياة الطيبة", "الروحية", "الاجتماعية",
    "العقلية", "الفكرية", "الجسدية", "البدنية", "العاطفية", "النفسية",
    "الإيمان", "التوكل", "الصبر", "الرضا", "الشكر", "الإحسان",
    "العدل", "الحكمة", "التوازن", "العبادة", "الأخلاق", "الاستقامة",
    "التقوى", "التزكية", "الالتزام", "الذكر", "التفكر", "الخشوع",
    "الورع", "الزهد", "القناعة", "التواضع", "الصدق", "الأمانة",
    "الإخلاص", "العفة", "الحياء", "الخوف", "الرجاء", "المحبة"
]

# Pillar keywords
PILLARS = {
    "spiritual": ["روحي", "روحية", "إيمان", "عبادة", "تقوى", "ذكر", "خشوع", "توكل", "الروحية"],
    "emotional": ["عاطفي", "عاطفية", "نفسي", "نفسية", "شعور", "سكينة", "طمأنينة", "العاطفية"],
    "intellectual": ["فكري", "فكرية", "عقلي", "عقلية", "تفكر", "حكمة", "علم", "الفكرية"],
    "physical": ["جسدي", "جسدية", "بدني", "بدنية", "صحة", "جسم", "البدنية"],
    "social": ["اجتماعي", "اجتماعية", "علاقات", "أسرة", "مجتمع", "تواصل", "الاجتماعية"],
}

# 30 depth-focused questions
QUESTIONS = [
    # 10 synthesis (global/world-model)
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


def count_pillars(text: str) -> int:
    """Count distinct pillars mentioned in response."""
    found = set()
    for pillar, keywords in PILLARS.items():
        if any(kw in text for kw in keywords):
            found.add(pillar)
    return len(found)


def count_concepts(text: str) -> int:
    """Count framework concepts in response."""
    return sum(1 for c in CONCEPTS if c in text)


def has_structure(text: str) -> bool:
    """Check if response has structured formatting."""
    has_bullets = bool(re.search(r'^\s*[-*•]\s', text, re.MULTILINE))
    has_numbers = bool(re.search(r'^\s*\d+[.)]\s', text, re.MULTILINE))
    has_headers = bool(re.search(r'^#+\s|^\*\*[^*]+\*\*:', text, re.MULTILINE))
    return has_bullets or has_numbers or has_headers


def has_summary(text: str) -> bool:
    """Check if response has a summary/conclusion."""
    markers = ["خلاصة", "ملخص", "في الختام", "توصيات", "نستنتج", "الخلاصة"]
    return any(m in text for m in markers)


def has_cross_pillar_connections(text: str) -> int:
    """Count explicit cross-pillar connections."""
    patterns = [
        r"العلاقة بين .* و",
        r"يؤثر .* على",
        r"يرتبط .* ب",
        r"الترابط بين",
        r"التكامل بين",
    ]
    count = 0
    for p in patterns:
        count += len(re.findall(p, text))
    return count


@dataclass
class QuestionResult:
    qid: str
    category: str
    success: bool
    words: int = 0
    concepts: int = 0
    pillars: int = 0
    connections: int = 0
    has_structure: bool = False
    has_summary: bool = False
    latency_ms: int = 0
    error: str = ""


@dataclass
class ModelStats:
    deployment: str
    results: list = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0
        return sum(1 for r in self.results if r.success) / len(self.results) * 100
    
    @property
    def avg_words(self) -> float:
        ok = [r for r in self.results if r.success]
        return sum(r.words for r in ok) / max(1, len(ok))
    
    @property
    def avg_concepts(self) -> float:
        ok = [r for r in self.results if r.success]
        return sum(r.concepts for r in ok) / max(1, len(ok))
    
    @property
    def avg_pillars(self) -> float:
        ok = [r for r in self.results if r.success]
        return sum(r.pillars for r in ok) / max(1, len(ok))
    
    @property
    def avg_connections(self) -> float:
        ok = [r for r in self.results if r.success]
        return sum(r.connections for r in ok) / max(1, len(ok))
    
    @property
    def structure_rate(self) -> float:
        ok = [r for r in self.results if r.success]
        return sum(1 for r in ok if r.has_structure) / max(1, len(ok)) * 100
    
    @property
    def summary_rate(self) -> float:
        ok = [r for r in self.results if r.success]
        return sum(1 for r in ok if r.has_summary) / max(1, len(ok)) * 100
    
    @property
    def multi_pillar_rate(self) -> float:
        """Rate of responses covering 3+ pillars."""
        ok = [r for r in self.results if r.success]
        return sum(1 for r in ok if r.pillars >= 3) / max(1, len(ok)) * 100
    
    def depth_score(self) -> float:
        """Score based on words, concepts, structure, summary."""
        return (
            min(self.avg_words / 400, 1) * 25 +
            min(self.avg_concepts / 10, 1) * 25 +
            (self.structure_rate / 100) * 25 +
            (self.summary_rate / 100) * 25
        )
    
    def cross_pillar_score(self) -> float:
        """Score based on pillars, connections, multi-pillar rate."""
        return (
            min(self.avg_pillars / 4, 1) * 35 +
            (self.multi_pillar_rate / 100) * 35 +
            min(self.avg_connections / 3, 1) * 30
        )
    
    def composite_score(self) -> float:
        """Weighted composite: 45% depth + 35% cross-pillar + 20% reliability."""
        return (
            self.depth_score() * 0.45 +
            self.cross_pillar_score() * 0.35 +
            self.success_rate * 0.20
        )


def extract_content(response) -> str:
    """Robustly extract content from response."""
    raw = response.choices[0].message.content
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", "") or item.get("content", ""))
        return "".join(parts).strip()
    return str(raw).strip()


def run_bakeoff():
    print("=" * 70)
    print("DIRECT SYSTEM BAKEOFF - Muhasibi Prompts")
    print("=" * 70)
    print(f"Models: {DEPLOYMENTS}")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Using: Muhasibi System Prompt (identical for all models)")
    print()
    
    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
        timeout=120
    )
    
    all_stats = []
    
    for dep in DEPLOYMENTS:
        print(f"\n{'-'*70}")
        print(f"TESTING: {dep}")
        print(f"{'-'*70}")
        
        stats = ModelStats(deployment=dep)
        
        for i, q in enumerate(QUESTIONS):
            print(f"  [{i+1}/{len(QUESTIONS)}] {q['id']}: {q['q'][:35]}... ", end="", flush=True)
            
            t0 = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=dep,
                    messages=[
                        {"role": "system", "content": MUHASIBI_SYSTEM_PROMPT},
                        {"role": "user", "content": q["q"]}
                    ],
                    max_completion_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    seed=SEED,
                    timeout=120
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)
                
                content = extract_content(response)
                
                result = QuestionResult(
                    qid=q["id"],
                    category=q["cat"],
                    success=len(content) > 0,
                    words=len(content.split()),
                    concepts=count_concepts(content),
                    pillars=count_pillars(content),
                    connections=has_cross_pillar_connections(content),
                    has_structure=has_structure(content),
                    has_summary=has_summary(content),
                    latency_ms=latency_ms,
                )
                stats.results.append(result)
                
                print(f"OK ({latency_ms}ms) w={result.words} c={result.concepts} p={result.pillars}")
                
            except Exception as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                result = QuestionResult(
                    qid=q["id"],
                    category=q["cat"],
                    success=False,
                    latency_ms=latency_ms,
                    error=str(e)[:100]
                )
                stats.results.append(result)
                print(f"ERROR: {str(e)[:50]}")
        
        all_stats.append(stats)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    ranked = sorted(all_stats, key=lambda s: -s.composite_score())
    
    print(f"\n{'Model':<15} {'Composite':>10} {'Depth':>8} {'Cross':>8} {'Success':>8} {'Words':>8} {'Concepts':>9} {'Pillars':>8} {'Connections':>12}")
    print("-" * 110)
    for s in ranked:
        print(f"{s.deployment:<15} {s.composite_score():>10.1f} {s.depth_score():>8.1f} {s.cross_pillar_score():>8.1f} {s.success_rate:>7.0f}% {s.avg_words:>8.0f} {s.avg_concepts:>9.1f} {s.avg_pillars:>8.2f} {s.avg_connections:>12.1f}")
    
    winner = ranked[0]
    print(f"\n*** WINNER: {winner.deployment} (composite={winner.composite_score():.1f}) ***")
    
    # Write report
    Path("eval/reports").mkdir(parents=True, exist_ok=True)
    with open("eval/reports/model_bakeoff_depth.md", "w", encoding="utf-8") as f:
        f.write("# Direct System Bakeoff Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Configuration\n\n")
        f.write("- **Method**: Direct LLM calls with Muḥāsibī system prompt\n")
        f.write("- **Control Variables**: seed=1337, temp=0.1, max_tokens=2000\n")
        f.write(f"- **Questions**: {len(QUESTIONS)} depth-focused questions\n")
        f.write("- **Scoring**: 45% Depth + 35% Cross-pillar + 20% Reliability\n\n")
        f.write("## Results\n\n")
        f.write("| Model | Composite | Depth | Cross-Pillar | Success% | Avg Words | Avg Concepts | Avg Pillars | Connections |\n")
        f.write("|-------|-----------|-------|--------------|----------|-----------|--------------|-------------|-------------|\n")
        for s in ranked:
            f.write(f"| {s.deployment} | {s.composite_score():.1f} | {s.depth_score():.1f} | {s.cross_pillar_score():.1f} | {s.success_rate:.0f}% | {s.avg_words:.0f} | {s.avg_concepts:.1f} | {s.avg_pillars:.2f} | {s.avg_connections:.1f} |\n")
        f.write(f"\n## Winner\n\n**{winner.deployment}** with composite score {winner.composite_score():.1f}\n")
        
        f.write("\n## Metrics Explained\n\n")
        f.write("- **Depth Score**: Words (25%) + Concepts (25%) + Structure (25%) + Summary (25%)\n")
        f.write("- **Cross-Pillar Score**: Pillars covered (35%) + Multi-pillar rate (35%) + Connections (30%)\n")
        f.write("- **Composite**: 45% Depth + 35% Cross-pillar + 20% Success rate\n")
        
        f.write("\n## Detailed Breakdown\n\n")
        for s in ranked:
            f.write(f"### {s.deployment}\n\n")
            f.write(f"- Success rate: {s.success_rate:.0f}%\n")
            f.write(f"- Avg words: {s.avg_words:.0f}\n")
            f.write(f"- Avg concepts: {s.avg_concepts:.1f}\n")
            f.write(f"- Avg pillars: {s.avg_pillars:.2f}\n")
            f.write(f"- Avg connections: {s.avg_connections:.1f}\n")
            f.write(f"- Structure rate: {s.structure_rate:.0f}%\n")
            f.write(f"- Summary rate: {s.summary_rate:.0f}%\n")
            f.write(f"- Multi-pillar rate (3+): {s.multi_pillar_rate:.0f}%\n\n")
    
    print(f"\nReport: eval/reports/model_bakeoff_depth.md")


if __name__ == "__main__":
    run_bakeoff()


