"""Export stakeholder FULL_SYSTEM outputs to a readable UTF-8 report.

Reason: Windows terminals often mangle Arabic/RTL + JSON; writing a UTF‑8 markdown
file ensures the outputs are readable and shareable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def infer_mode(answer_ar: str, abstained: bool) -> str:
    if abstained:
        return "abstained"
    t = answer_ar or ""
    if "قسم (أ): ما يمكن دعمه من الأدلة المسترجعة" in t:
        return "partial_scenario"
    if "مصفوفة المقارنة" in t:
        return "compare"
    if ("الربط بين الركائز" in t) and ("تنبيهات" in t) and ("خلاصة تنفيذية" in t):
        return "deep"
    return "light_deep"


def _md_escape(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    # Pick latest stakeholder FULL_SYSTEM output (avoid hardcoding run_id).
    out_dir = Path("eval/output")
    # Pick latest stakeholder FULL_SYSTEM output across versions (v1/v2...).
    candidates = sorted(out_dir.glob("stakeholder_acceptance__*__FULL_SYSTEM.jsonl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit("No stakeholder FULL_SYSTEM outputs found under eval/output/")
    in_path = candidates[-1]
    rows = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    out_path = Path("eval/reports/stakeholder_acceptance_outputs.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("## Stakeholder acceptance outputs (FULL_SYSTEM)")
    lines.append("")
    lines.append(f"- Source JSONL: `{in_path.as_posix()}`")
    lines.append(f"- Rows: {len(rows)}")
    lines.append("")

    for i, r in enumerate(rows, start=1):
        rid = str(r.get("id") or "")
        q = str(r.get("question") or "")
        abstained = bool(r.get("abstained"))
        mode_used = infer_mode(str(r.get("answer_ar") or ""), abstained)
        abstain_reason = str(r.get("abstain_reason") or "")

        lines.append(f"### {i}) {rid}")
        lines.append("")
        lines.append(f"- **mode_used**: `{mode_used}`")
        lines.append(f"- **abstained**: `{abstained}`")
        if abstained:
            lines.append(f"- **abstain_reason**: `{abstain_reason}`")
        lines.append("")
        lines.append("**question_ar**")
        lines.append("")
        lines.append(_md_escape(q))
        lines.append("")
        lines.append("**answer_ar**")
        lines.append("")
        lines.append("```")
        lines.append(_md_escape(str(r.get("answer_ar") or "")))
        lines.append("```")
        lines.append("")
        lines.append("**citations (with quotes)**")
        lines.append("")
        cits = r.get("citations") or []
        if not cits:
            lines.append("- (none)")
        else:
            for c in cits:
                source_id = str((c or {}).get("source_id") or "")
                span_start = (c or {}).get("span_start")
                span_end = (c or {}).get("span_end")
                quote = str((c or {}).get("quote") or "")
                lines.append(f"- `{source_id}` [{span_start}:{span_end}] — {quote}")
        lines.append("")
        lines.append("**graph_trace**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(r.get("graph_trace") or {}, ensure_ascii=False, sort_keys=True))
        lines.append("```")
        lines.append("")

        if r.get("debug", {}).get("contract") is not None:
            lines.append("**contract (debug)**")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(r.get("debug", {}).get("contract") or {}, ensure_ascii=False, sort_keys=True))
            lines.append("```")
            lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
