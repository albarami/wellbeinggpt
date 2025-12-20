"""Coverage Audit Script.

Moved implementation into smaller modules to enforce the max-500-LOC rule.

Usage:
  python -m scripts.coverage_audit
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from scripts.coverage_audit_impl import require_db, run_coverage_audit


def main() -> None:
    load_dotenv()
    require_db()

    print("Running coverage audit...")
    print("=" * 60)

    report = asyncio.run(run_coverage_audit())

    output_path = Path("coverage_report.json")
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Report written to: {output_path}")
    print()
    print("SUMMARY:")
    print(f"  Status: {report['status']}")
    print(f"  Doc file: {report.get('doc_file')}")
    print(f"  Doc hash: {report.get('doc_hash')}")

    print()
    print("COUNTS:")
    for k, v in report.get("counts", {}).items():
        print(f"  {k}: {v}")

    print()
    print("EDGE COUNTS:")
    for k, v in report.get("edge_counts", {}).items():
        print(f"  {k}: {v}")

    total_missing = sum(len(v) for v in report.get("missing", {}).values())
    if total_missing > 0:
        print(f"\nMISSING ITEMS: {total_missing} total")
        for category, items in report.get("missing", {}).items():
            if items:
                print(f"  {category}: {len(items)} missing")
        sys.exit(1)

    print("\nMISSING ITEMS: None - all complete!")
    sys.exit(0)


if __name__ == "__main__":
    main()
