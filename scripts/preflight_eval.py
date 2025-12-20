"""Preflight checks for evaluation runs.

Goal:
- Fail fast (within seconds) when the environment is likely to cause
  `Command failed to spawn: Aborted` on Windows.

Design:
- Deterministic, no external dependencies (no psutil).
- Works on Windows via `tasklist` (CSV output).
- Safe by default: does not kill anything; only reports and can fail with exit code.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from io import StringIO
from typing import Iterable


@dataclass(frozen=True)
class PythonProcess:
    pid: int
    mem_kb: int


def parse_tasklist_csv(text: str) -> list[PythonProcess]:
    """Parse `tasklist /FO CSV` output into PythonProcess list.

    Args:
        text: Raw CSV text from `tasklist`.

    Returns:
        List of python.exe processes with PID + memory (KB).
    """
    if not text.strip():
        return []

    rdr = csv.DictReader(StringIO(text))
    out: list[PythonProcess] = []
    for row in rdr:
        try:
            name = str(row.get("Image Name") or "")
            if name.lower() != "python.exe":
                continue
            pid = int(str(row.get("PID") or "0").strip() or "0")
            mem_raw = str(row.get("Mem Usage") or "").strip()
            # Example: "5,888,004 K"
            mem_raw = mem_raw.replace(",", "").replace("K", "").strip()
            mem_kb = int(mem_raw.split()[0]) if mem_raw else 0
            if pid > 0:
                out.append(PythonProcess(pid=pid, mem_kb=max(0, mem_kb)))
        except Exception:
            continue
    return out


def _run_tasklist() -> str:
    cp = subprocess.run(
        ["tasklist", "/FO", "CSV"],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="ignore",
    )
    return cp.stdout or ""


def list_python_processes() -> list[PythonProcess]:
    """List running python processes (Windows only)."""
    if os.name != "nt":
        return []
    return parse_tasklist_csv(_run_tasklist())


def preflight_check(
    *,
    max_python_processes: int,
    max_total_mem_mb: int,
) -> tuple[bool, str]:
    """Run preflight and return (ok, message)."""
    if os.name != "nt":
        return True, "Non-Windows OS: preflight skipped."

    procs = list_python_processes()
    total_kb = sum(p.mem_kb for p in procs)
    total_mb = int(total_kb / 1024)

    msg_lines: list[str] = []
    msg_lines.append(f"python.exe count: {len(procs)} (max allowed: {max_python_processes})")
    msg_lines.append(f"python.exe total mem: {total_mb} MB (max allowed: {max_total_mem_mb} MB)")

    worst = sorted(procs, key=lambda p: p.mem_kb, reverse=True)[:10]
    if worst:
        msg_lines.append("Top python.exe processes by memory:")
        for p in worst:
            msg_lines.append(f"  - PID {p.pid}: {int(p.mem_kb/1024)} MB")

    ok = True
    if len(procs) > max_python_processes:
        ok = False
        msg_lines.append("FAIL: too many python.exe processes (likely to cause spawn abort).")
    if total_mb > max_total_mem_mb:
        ok = False
        msg_lines.append("FAIL: python.exe memory too high (likely to cause spawn abort).")

    if not ok:
        msg_lines.append("Action: stop old eval runs / notebooks, then retry.")
        msg_lines.append("Tip: `taskkill /PID <pid> /F` for a stuck PID (use carefully).")

    return ok, "\n".join(msg_lines)


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-python-processes", type=int, default=8)
    p.add_argument("--max-total-mem-mb", type=int, default=4096)
    p.add_argument("--fail", action="store_true", help="Exit non-zero if limits exceeded.")
    args = p.parse_args(list(argv) if argv is not None else None)

    ok, msg = preflight_check(
        max_python_processes=int(args.max_python_processes),
        max_total_mem_mb=int(args.max_total_mem_mb),
    )
    print(msg)
    if args.fail and not ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

