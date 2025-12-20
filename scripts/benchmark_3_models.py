"""
Standalone benchmark comparing GPT-5 / GPT-5.1 / GPT-5.2 deployments.

No pytest, no heavy DB ingestion. Just direct Azure OpenAI calls.

Usage:
    python scripts/benchmark_3_models.py
"""

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Load .env
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip()

DEPLOYMENTS = os.getenv("MODEL_VARIANT_DEPLOYMENTS", "gpt-5-chat,gpt-5.1,gpt-5.2").split(",")
DEPLOYMENTS = [d.strip() for d in DEPLOYMENTS if d.strip()]

if not ENDPOINT or not API_KEY:
    print("ERROR: AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY not set in .env")
    sys.exit(1)

if len(DEPLOYMENTS) < 3:
    print(f"ERROR: Need 3 deployments, got {DEPLOYMENTS}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Questions (Arabic, wellbeing-focused)
# ─────────────────────────────────────────────────────────────────────────────
QUESTIONS = [
    ("definition", "ما هو الإيمان؟"),
    ("comparison", 'ما الفرق بين الصبر والرضا؟'),
    ("cross_pillar", "كيف ترتبط الحياة الروحية بالحياة الاجتماعية؟"),
    ("synthesis", "كيف تعمل الركائز الخمس معًا لتحقيق الحياة الطيبة؟"),
    ("practical", "كيف أطبق الصبر عند الغضب؟"),
    ("out_of_scope", "ما هي نظرية الكم في الفيزياء؟"),
]

SYSTEM_PROMPT = """أنت مساعد متخصص في إطار الحياة الطيبة (الركائز الخمس).
أجب بالعربية فقط. إذا كان السؤال خارج نطاق الإطار، قل "خارج نطاق الإطار".
كن مختصرًا ودقيقًا."""


@dataclass
class ModelStats:
    deployment: str
    questions_answered: int = 0
    out_of_scope_detected: int = 0
    errors: int = 0
    total_latency_s: float = 0.0
    responses: list = field(default_factory=list)

    def avg_latency(self) -> float:
        n = self.questions_answered + self.out_of_scope_detected + self.errors
        return self.total_latency_s / max(1, n)

    def score(self) -> float:
        """Higher is better: answered questions + fast response."""
        return self.questions_answered * 10 - self.errors * 20 - self.avg_latency() * 2


def run_benchmark():
    print("=" * 70)
    print("3-MODEL BENCHMARK: GPT-5 vs GPT-5.1 vs GPT-5.2")
    print("=" * 70)
    print(f"Endpoint: {ENDPOINT}")
    print(f"API Version: {API_VERSION}")
    print(f"Deployments: {DEPLOYMENTS}")
    print(f"Questions: {len(QUESTIONS)}")
    print()

    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
    )

    all_stats: list[ModelStats] = []

    for dep in DEPLOYMENTS:
        print(f"\n{'-' * 70}")
        print(f"TESTING: {dep}")
        print(f"{'-' * 70}")

        stats = ModelStats(deployment=dep)

        for q_name, q_text in QUESTIONS:
            print(f"  [{q_name}] {q_text[:40]}... ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=dep,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": q_text},
                    ],
                    max_completion_tokens=300,
                    timeout=60,
                )
                dt = time.perf_counter() - t0
                stats.total_latency_s += dt

                content = (response.choices[0].message.content or "").strip()
                stats.responses.append((q_name, content[:200]))

                if "خارج نطاق" in content or "outside" in content.lower():
                    stats.out_of_scope_detected += 1
                    print(f"OOS ({dt:.1f}s)")
                else:
                    stats.questions_answered += 1
                    print(f"OK ({dt:.1f}s) -> {content[:50]}...")

            except Exception as e:
                dt = time.perf_counter() - t0
                stats.total_latency_s += dt
                stats.errors += 1
                print(f"ERROR ({dt:.1f}s): {e}")

        all_stats.append(stats)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n")
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Sort by score descending
    ranked = sorted(all_stats, key=lambda s: -s.score())

    for i, s in enumerate(ranked, 1):
        print(f"\n#{i} {s.deployment}")
        print(f"    Score: {s.score():.1f}")
        print(f"    Answered: {s.questions_answered}/{len(QUESTIONS)}")
        print(f"    Out-of-scope detected: {s.out_of_scope_detected}")
        print(f"    Errors: {s.errors}")
        print(f"    Avg latency: {s.avg_latency():.2f}s")

    print("\n" + "=" * 70)
    print(f"WINNER: {ranked[0].deployment} (score={ranked[0].score():.1f})")
    print("=" * 70)

    return ranked


if __name__ == "__main__":
    run_benchmark()
