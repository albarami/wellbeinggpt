"""Generate deep cross-pillar dataset (deterministic).

Creates `eval/datasets/deep_cross_pillar_gold.jsonl` from the existing
`eval/datasets/cross_pillar.jsonl` by adding scholar-format requirements.

No DB access and no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    src = Path("eval/datasets/cross_pillar.jsonl")
    dst = Path("eval/datasets/deep_cross_pillar_gold.jsonl")

    if not src.exists():
        raise SystemExit(f"Missing source dataset: {src}")

    rows = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))

    # Deterministic slice: first 60.
    base = rows[:60]
    out_rows = []
    for i, r in enumerate(base, start=1):
        rid = f"deepcross-{i:04d}"
        q = str(r.get("question_ar") or "")
        q = (
            q
            + "\n\n"
            + "اكتب الإجابة بصيغة (عالم/باحث) داخل إطار الحياة الطيبة، وبالترتيب التالي: "
            + "تعريف المفهوم داخل الإطار؛ التأصيل والأدلة (مختصر ومركز)؛ الربط بين الركائز (مع سبب الربط)؛ "
            + "تطبيق عملي على الحالة/السؤال؛ تنبيهات وأخطاء شائعة؛ خلاصة تنفيذية (3 نقاط). "
            + "كل قسم يجب أن يكون مدعومًا باستشهادات."
        )

        rr = dict(r)
        rr["id"] = rid
        rr["question_ar"] = q
        rr["tags"] = list(dict.fromkeys((rr.get("tags") or []) + ["deep"]))

        ar = dict(rr.get("answer_requirements") or {})
        ar["format"] = "scholar"
        ar["must_include"] = list(
            dict.fromkeys(
                (ar.get("must_include") or [])
                + [
                    "تعريف المفهوم داخل الإطار",
                    "التأصيل والأدلة",
                    "الربط بين الركائز",
                    "تطبيق عملي",
                    "تنبيهات وأخطاء شائعة",
                    "خلاصة تنفيذية",
                ]
            )
        )
        rr["answer_requirements"] = ar

        out_rows.append(rr)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
