"""
Accelerated Trace Collection Script.

Runs bakeoff datasets repeatedly to quickly collect training traces.
Monitors progress and stops when target is reached.

Usage:
    python scripts/accelerate_trace_collection.py [--target 500]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
TRACE_DIR = PROJECT_ROOT / "data" / "phase2" / "edge_traces" / "train"
BAKEOFF_160 = PROJECT_ROOT / "eval" / "datasets" / "bakeoff_depth_v1.jsonl"
SANITY_50 = PROJECT_ROOT / "eval" / "datasets" / "bakeoff_sanity_gate_v1.jsonl"

API_URL = "http://localhost:8002/ask/ui"
TIMEOUT = 120.0


def count_traces() -> int:
    """Count current training traces."""
    if not TRACE_DIR.exists():
        return 0
    count = 0
    for trace_file in TRACE_DIR.glob("*.jsonl"):
        with open(trace_file, "r", encoding="utf-8") as f:
            count += sum(1 for _ in f)
    return count


def load_questions(dataset_path: Path) -> list[str]:
    """Load questions from a JSONL dataset."""
    questions = []
    if not dataset_path.exists():
        return questions
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                q = item.get("question", "")
                if q:
                    questions.append(q)
            except json.JSONDecodeError:
                pass
    return questions


def send_question(client: httpx.Client, question: str) -> str:
    """Send a question and return contract outcome."""
    try:
        response = client.post(
            API_URL,
            json={"question": question, "mode": "answer"},
            timeout=TIMEOUT,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("contract", {}).get("outcome", "UNKNOWN")
        return f"HTTP_{response.status_code}"
    except Exception as e:
        return f"ERROR: {str(e)[:50]}"


def run_collection(target_traces: int = 500, audit_interval: int = 50):
    """Run accelerated trace collection."""
    logger.info("=" * 70)
    logger.info("ACCELERATED TRACE COLLECTION")
    logger.info("=" * 70)
    
    initial_count = count_traces()
    logger.info(f"\nInitial trace count: {initial_count}")
    logger.info(f"Target: {target_traces} traces")
    logger.info(f"Need: {max(0, target_traces - initial_count)} more traces")
    
    if initial_count >= target_traces:
        logger.info("Target already reached!")
        return
    
    # Load questions
    questions_160 = load_questions(BAKEOFF_160)
    questions_50 = load_questions(SANITY_50)
    
    logger.info(f"\nLoaded datasets:")
    logger.info(f"  bakeoff_depth_v1: {len(questions_160)} questions")
    logger.info(f"  bakeoff_sanity_gate_v1: {len(questions_50)} questions")
    
    # Combine and dedupe
    all_questions = list(set(questions_160 + questions_50))
    logger.info(f"  Total unique questions: {len(all_questions)}")
    
    # Run collection
    logger.info(f"\nStarting collection...")
    logger.info("-" * 50)
    
    outcomes = {"PASS_FULL": 0, "PASS_PARTIAL": 0, "FAIL": 0, "OTHER": 0}
    round_num = 0
    
    with httpx.Client() as client:
        while count_traces() < target_traces:
            round_num += 1
            logger.info(f"\n[Round {round_num}] Processing {len(all_questions)} questions...")
            
            for i, question in enumerate(all_questions, 1):
                outcome = send_question(client, question)
                
                if outcome == "PASS_FULL":
                    outcomes["PASS_FULL"] += 1
                elif outcome == "PASS_PARTIAL":
                    outcomes["PASS_PARTIAL"] += 1
                elif outcome == "FAIL":
                    outcomes["FAIL"] += 1
                else:
                    outcomes["OTHER"] += 1
                
                # Progress update every 10 questions
                if i % 10 == 0:
                    current = count_traces()
                    logger.info(f"  [{i}/{len(all_questions)}] traces={current} | PASS_FULL={outcomes['PASS_FULL']}")
                
                # Check target
                if count_traces() >= target_traces:
                    logger.info(f"\n  TARGET REACHED!")
                    break
                
                # Small delay to avoid overwhelming
                time.sleep(0.1)
            
            # Audit checkpoint
            current = count_traces()
            if current >= initial_count + audit_interval:
                logger.info(f"\n  [CHECKPOINT] {current} traces collected")
                # Run quick audit
                try:
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, "scripts/audit_edge_traces.py"],
                        capture_output=True,
                        text=True,
                        cwd=str(PROJECT_ROOT),
                    )
                    # Extract key metrics from output
                    for line in result.stdout.split("\n"):
                        if "Status:" in line or "Positive rate" in line or "Separation" in line:
                            logger.info(f"    {line.strip()}")
                except Exception:
                    pass
    
    # Final summary
    final_count = count_traces()
    logger.info(f"\n" + "=" * 70)
    logger.info("COLLECTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total traces: {final_count}")
    logger.info(f"New traces: {final_count - initial_count}")
    logger.info(f"Outcomes: {outcomes}")
    
    if final_count >= target_traces:
        logger.info("\nNext steps:")
        logger.info("  1. Run: python scripts/audit_edge_traces.py")
        logger.info("  2. If audit passes, revert EDGE_TRACE_SAMPLE_RATE=0.25")
        logger.info("  3. Continue collecting to 2,000 traces")


def main():
    parser = argparse.ArgumentParser(description="Accelerate trace collection")
    parser.add_argument("--target", type=int, default=500, help="Target trace count")
    parser.add_argument("--audit-interval", type=int, default=50, help="Audit every N traces")
    args = parser.parse_args()
    
    run_collection(target_traces=args.target, audit_interval=args.audit_interval)


if __name__ == "__main__":
    main()
