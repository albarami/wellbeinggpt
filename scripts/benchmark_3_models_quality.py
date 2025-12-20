"""
Quality-focused benchmark comparing GPT-5 / GPT-5.1 / GPT-5.2.

Evaluates:
1. Response length (detail)
2. Arabic fluency markers
3. Structure (bullet points, numbered lists)
4. Mentions of framework concepts
5. Speed

Usage:
    python scripts/benchmark_3_models_quality.py
"""

import os
import sys
import time
import re
from dataclasses import dataclass, field
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

# Config
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip()

DEPLOYMENTS = os.getenv("MODEL_VARIANT_DEPLOYMENTS", "gpt-5-chat,gpt-5.1,gpt-5.2").split(",")
DEPLOYMENTS = [d.strip() for d in DEPLOYMENTS if d.strip()]

# Framework concepts to check for
FRAMEWORK_CONCEPTS = [
    "الركيزة", "الركائز", "الحياة الطيبة", "الروحية", "الاجتماعية", 
    "العقلية", "الجسدية", "العاطفية", "الإيمان", "التوكل", "الصبر",
    "الرضا", "الشكر", "الإحسان", "العدل", "الحكمة", "التوازن"
]

QUESTIONS = [
    {
        "name": "deep_definition",
        "text": "اشرح مفهوم الإيمان في إطار الحياة الطيبة بالتفصيل مع ذكر علاقته بالركائز الأخرى",
        "expect_long": True,
    },
    {
        "name": "compare_analyze",
        "text": "قارن بين الصبر والرضا من حيث التعريف والأهمية والتطبيق العملي",
        "expect_long": True,
    },
    {
        "name": "cross_pillar_deep",
        "text": "كيف تتكامل الحياة الروحية مع الحياة الاجتماعية والعقلية لتحقيق التوازن؟ اذكر أمثلة عملية.",
        "expect_long": True,
    },
    {
        "name": "practical_steps",
        "text": "أريد خطة عملية مفصلة لتطبيق الصبر في حياتي اليومية. اذكر خطوات محددة.",
        "expect_long": True,
    },
    {
        "name": "synthesis_all",
        "text": "اشرح كيف تعمل الركائز الخمس معًا كمنظومة متكاملة لتحقيق الحياة الطيبة. استخدم أمثلة.",
        "expect_long": True,
    },
]

SYSTEM_PROMPT = """أنت عالم متخصص في إطار الحياة الطيبة (الركائز الخمس).
مهمتك تقديم إجابات شاملة ومفصلة بالعربية الفصحى.
استخدم التنسيق المناسب (قوائم، نقاط، عناوين) لتوضيح الإجابة.
اربط المفاهيم ببعضها واذكر الأدلة والأمثلة العملية."""


@dataclass
class ResponseAnalysis:
    question_name: str
    content: str
    latency_s: float
    error: str = ""
    
    # Quality metrics
    word_count: int = 0
    has_structure: bool = False  # bullets, numbers, headers
    framework_mentions: int = 0
    arabic_quality_score: float = 0.0


@dataclass 
class ModelQualityStats:
    deployment: str
    responses: list = field(default_factory=list)
    total_latency_s: float = 0.0
    errors: int = 0
    
    def avg_word_count(self) -> float:
        valid = [r for r in self.responses if not r.error]
        return sum(r.word_count for r in valid) / max(1, len(valid))
    
    def avg_framework_mentions(self) -> float:
        valid = [r for r in self.responses if not r.error]
        return sum(r.framework_mentions for r in valid) / max(1, len(valid))
    
    def structure_rate(self) -> float:
        valid = [r for r in self.responses if not r.error]
        return sum(1 for r in valid if r.has_structure) / max(1, len(valid)) * 100
    
    def avg_latency(self) -> float:
        return self.total_latency_s / max(1, len(self.responses))
    
    def quality_score(self) -> float:
        """Composite quality score (higher = better)."""
        # Normalize components
        word_score = min(self.avg_word_count() / 200, 1.0) * 30  # Up to 30 pts for detail
        framework_score = min(self.avg_framework_mentions() / 5, 1.0) * 25  # Up to 25 pts
        structure_score = self.structure_rate() / 100 * 20  # Up to 20 pts
        error_penalty = self.errors * 10
        speed_bonus = max(0, 10 - self.avg_latency())  # Up to 10 pts for speed
        
        return word_score + framework_score + structure_score + speed_bonus - error_penalty


def analyze_response(content: str) -> tuple[int, bool, int]:
    """Analyze response for quality metrics."""
    # Word count (Arabic words)
    words = len(content.split())
    
    # Structure detection
    has_structure = bool(
        re.search(r'^\s*[-*•]\s', content, re.MULTILINE) or  # Bullets
        re.search(r'^\s*\d+[.)]\s', content, re.MULTILINE) or  # Numbers
        re.search(r'\*\*[^*]+\*\*', content) or  # Bold
        re.search(r'^#+\s', content, re.MULTILINE)  # Headers
    )
    
    # Framework concept mentions
    mentions = sum(1 for concept in FRAMEWORK_CONCEPTS if concept in content)
    
    return words, has_structure, mentions


def run_quality_benchmark():
    print("=" * 70)
    print("QUALITY BENCHMARK: GPT-5 vs GPT-5.1 vs GPT-5.2")
    print("=" * 70)
    print(f"Evaluating: Detail, Structure, Framework Knowledge, Speed")
    print(f"Questions: {len(QUESTIONS)} (complex, expecting detailed answers)")
    print()

    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
    )

    all_stats: list[ModelQualityStats] = []

    for dep in DEPLOYMENTS:
        print(f"\n{'-' * 70}")
        print(f"TESTING: {dep}")
        print(f"{'-' * 70}")

        stats = ModelQualityStats(deployment=dep)

        for q in QUESTIONS:
            print(f"  [{q['name']}] ... ", end="", flush=True)
            t0 = time.perf_counter()
            
            try:
                response = client.chat.completions.create(
                    model=dep,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": q["text"]},
                    ],
                    max_completion_tokens=800,
                    timeout=90,
                )
                dt = time.perf_counter() - t0
                stats.total_latency_s += dt

                content = (response.choices[0].message.content or "").strip()
                words, has_struct, mentions = analyze_response(content)
                
                analysis = ResponseAnalysis(
                    question_name=q["name"],
                    content=content,
                    latency_s=dt,
                    word_count=words,
                    has_structure=has_struct,
                    framework_mentions=mentions,
                )
                stats.responses.append(analysis)
                
                print(f"OK ({dt:.1f}s) words={words} struct={'Y' if has_struct else 'N'} concepts={mentions}")

            except Exception as e:
                dt = time.perf_counter() - t0
                stats.total_latency_s += dt
                stats.errors += 1
                stats.responses.append(ResponseAnalysis(
                    question_name=q["name"],
                    content="",
                    latency_s=dt,
                    error=str(e)[:50],
                ))
                print(f"ERROR ({dt:.1f}s): {e}")

        all_stats.append(stats)

    # Summary
    print("\n")
    print("=" * 70)
    print("QUALITY RESULTS")
    print("=" * 70)

    # Sort by quality score descending
    ranked = sorted(all_stats, key=lambda s: -s.quality_score())

    print(f"\n{'Model':<15} {'Quality':>8} {'Words':>8} {'Struct%':>8} {'Concepts':>8} {'Speed':>8} {'Errors':>6}")
    print("-" * 70)
    
    for s in ranked:
        print(f"{s.deployment:<15} {s.quality_score():>8.1f} {s.avg_word_count():>8.0f} {s.structure_rate():>7.0f}% {s.avg_framework_mentions():>8.1f} {s.avg_latency():>7.1f}s {s.errors:>6}")

    print("\n" + "=" * 70)
    winner = ranked[0]
    print(f"WINNER: {winner.deployment}")
    print(f"  Quality Score: {winner.quality_score():.1f}")
    print(f"  Avg Words: {winner.avg_word_count():.0f}")
    print(f"  Structure Use: {winner.structure_rate():.0f}%")
    print(f"  Framework Concepts: {winner.avg_framework_mentions():.1f} per answer")
    print(f"  Avg Speed: {winner.avg_latency():.1f}s")
    print("=" * 70)

    # Show sample responses from winner
    print(f"\n--- Sample Response from {winner.deployment} ---")
    if winner.responses:
        sample = winner.responses[0]
        print(f"Question: {QUESTIONS[0]['name']}")
        print(f"Response ({sample.word_count} words):")
        print(sample.content[:500] + "..." if len(sample.content) > 500 else sample.content)

    return ranked


if __name__ == "__main__":
    run_quality_benchmark()
