"""
Acceptance Suite Runner

Runs all coverage/quality checks and produces a single pass/fail verdict.

Usage:
    python scripts/run_acceptance_suite.py

This script:
1. Runs coverage_audit.py and fails if any missing.* is non-empty
2. Runs vector_coverage_check.py and fails if any entity fails top-K
3. Runs pytest and fails on any test failure
4. Writes acceptance_summary.json

Env vars required:
    DATABASE_URL - Postgres connection string
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY (optional)
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def run_command(cmd: list[str], description: str) -> tuple[bool, str]:
    """
    Run a command and return (success, output).
    """
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        output = result.stdout + result.stderr
        print(output)
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("ACCEPTANCE SUITE RUNNER")
    print(f"Started at: {datetime.utcnow().isoformat()}")
    print("=" * 60)
    
    summary: dict[str, Any] = {
        "started_at": datetime.utcnow().isoformat(),
        "checks": {},
        "overall_status": "pending",
    }
    
    all_passed = True
    
    # 1. Coverage audit
    success, output = run_command(
        [sys.executable, "scripts/coverage_audit.py"],
        "Coverage Audit"
    )
    summary["checks"]["coverage_audit"] = {
        "passed": success,
        "report_file": "coverage_report.json",
    }
    if not success:
        all_passed = False
        print("\n❌ Coverage audit FAILED")
    else:
        print("\n✓ Coverage audit PASSED")
    
    # 2. Vector coverage check (skip if Azure Search not configured)
    import os
    if os.getenv("AZURE_SEARCH_ENDPOINT") and os.getenv("AZURE_SEARCH_API_KEY"):
        success, output = run_command(
            [sys.executable, "scripts/vector_coverage_check.py", "--top-k", "10"],
            "Vector Coverage Check"
        )
        summary["checks"]["vector_coverage"] = {
            "passed": success,
            "report_file": "vector_coverage_report.json",
        }
        if not success:
            all_passed = False
            print("\n❌ Vector coverage check FAILED")
        else:
            print("\n✓ Vector coverage check PASSED")
    else:
        print("\n⚠ Vector coverage check SKIPPED (Azure Search not configured)")
        summary["checks"]["vector_coverage"] = {
            "passed": None,
            "skipped": True,
            "reason": "Azure Search not configured",
        }
    
    # 3. Pytest
    success, output = run_command(
        [sys.executable, "-m", "pytest", "-v", "--tb=short"],
        "Pytest Suite"
    )
    summary["checks"]["pytest"] = {
        "passed": success,
    }
    if not success:
        all_passed = False
        print("\n❌ Pytest FAILED")
    else:
        print("\n✓ Pytest PASSED")
    
    # Summary
    summary["completed_at"] = datetime.utcnow().isoformat()
    summary["overall_status"] = "passed" if all_passed else "failed"
    
    # Write summary
    output_path = Path("acceptance_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("ACCEPTANCE SUITE SUMMARY")
    print("=" * 60)
    print(f"Summary written to: {output_path}")
    print()
    
    for check_name, result in summary["checks"].items():
        if result.get("skipped"):
            status = "⚠ SKIPPED"
        elif result.get("passed"):
            status = "✓ PASSED"
        else:
            status = "❌ FAILED"
        print(f"  {check_name}: {status}")
    
    print()
    if all_passed:
        print("🎉 OVERALL: ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("💥 OVERALL: SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()




