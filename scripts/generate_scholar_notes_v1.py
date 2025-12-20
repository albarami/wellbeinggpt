"""Generate a minimal Scholar Notes v1 starter pack from existing chunks.

Design:
- Deterministic (no LLM).
- Framework-only: copies existing definition chunks.
- Every note includes >=1 evidence span bound to an existing chunk substring.

Usage:
  python -m scripts.generate_scholar_notes_v1 --out data/scholar_notes/notes_v1.jsonl --limit 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.ingest.sentence_spans import sentence_spans
from eval.datasets.source_loader import load_dotenv_if_present


@dataclass(frozen=True)
class _Candidate:
    entity_type: str
    pillar_id: str
    core_value_id: Optional[str]
    sub_value_id: Optional[str]
    title_ar: str
    chunk_id: str
    text_ar: str
    source_doc_id: str


def _first_span(text_ar: str) -> tuple[int, int, str]:
    spans = sentence_spans(text_ar or "", max_spans=64)
    if spans:
        s0 = spans[0]
        quote = (text_ar or "")[s0.start : s0.end]
        return int(s0.start), int(s0.end), quote
    # Fallback: first 200 chars (still a strict substring).
    end = min(200, len(text_ar or ""))
    quote = (text_ar or "")[:end]
    return 0, end, quote


async def _fetch_candidates(*, limit: int) -> list[_Candidate]:
    q_sub = text(
        """
        SELECT
          'sub_value' AS entity_type,
          cv.pillar_id AS pillar_id,
          sv.core_value_id AS core_value_id,
          sv.id AS sub_value_id,
          sv.name_ar AS title_ar,
          c.chunk_id AS chunk_id,
          c.text_ar AS text_ar,
          c.source_doc_id AS source_doc_id
        FROM sub_value sv
        JOIN core_value cv ON cv.id = sv.core_value_id
        JOIN chunk c ON c.entity_type='sub_value' AND c.entity_id=sv.id AND c.chunk_type='definition'
        WHERE c.text_ar IS NOT NULL AND c.text_ar <> '' AND length(c.text_ar) >= 120
        ORDER BY c.chunk_id
        LIMIT :lim
        """
    )
    q_core = text(
        """
        SELECT
          'core_value' AS entity_type,
          cv.pillar_id AS pillar_id,
          cv.id AS core_value_id,
          NULL AS sub_value_id,
          cv.name_ar AS title_ar,
          c.chunk_id AS chunk_id,
          c.text_ar AS text_ar,
          c.source_doc_id AS source_doc_id
        FROM core_value cv
        JOIN chunk c ON c.entity_type='core_value' AND c.entity_id=cv.id AND c.chunk_type='definition'
        WHERE c.text_ar IS NOT NULL AND c.text_ar <> '' AND length(c.text_ar) >= 120
        ORDER BY c.chunk_id
        LIMIT :lim
        """
    )
    q_pillar = text(
        """
        SELECT
          'pillar' AS entity_type,
          p.id AS pillar_id,
          NULL AS core_value_id,
          NULL AS sub_value_id,
          p.name_ar AS title_ar,
          c.chunk_id AS chunk_id,
          c.text_ar AS text_ar,
          c.source_doc_id AS source_doc_id
        FROM pillar p
        JOIN chunk c ON c.entity_type='pillar' AND c.entity_id=p.id AND c.chunk_type='definition'
        WHERE c.text_ar IS NOT NULL AND c.text_ar <> '' AND length(c.text_ar) >= 120
        ORDER BY c.chunk_id
        LIMIT :lim
        """
    )

    async with get_session() as session:
        out: list[_Candidate] = []
        for q in (q_sub, q_core, q_pillar):
            rows = (await session.execute(q, {"lim": int(limit)})).fetchall()
            for r in rows:
                out.append(
                    _Candidate(
                        entity_type=str(r.entity_type),
                        pillar_id=str(r.pillar_id),
                        core_value_id=str(r.core_value_id) if r.core_value_id is not None else None,
                        sub_value_id=str(r.sub_value_id) if r.sub_value_id is not None else None,
                        title_ar=str(r.title_ar or "").strip(),
                        chunk_id=str(r.chunk_id or "").strip(),
                        text_ar=str(r.text_ar or ""),
                        source_doc_id=str(r.source_doc_id or "").strip(),
                    )
                )

    # Deduplicate by chunk_id deterministically.
    seen: set[str] = set()
    uniq: list[_Candidate] = []
    for c in sorted(out, key=lambda x: (x.entity_type, x.chunk_id)):
        if not c.chunk_id or c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        uniq.append(c)
    return uniq[: max(0, int(limit))]


def _note_row(c: _Candidate, *, version: str) -> dict[str, Any]:
    s, e, q = _first_span(c.text_ar)
    return {
        "note_id": f"SNV1_{c.entity_type}_{c.chunk_id}",
        "pillar_id": c.pillar_id,
        "core_value_id": c.core_value_id,
        "sub_value_id": c.sub_value_id,
        "title_ar": c.title_ar,
        "definition_ar": c.text_ar,
        "deep_explanation_ar": "",
        "cross_pillar_links": [],
        "applied_scenarios": [],
        "common_misunderstandings": [],
        "evidence_spans": [
            {
                "source_id": c.source_doc_id,
                "chunk_id": c.chunk_id,
                "span_start": int(s),
                "span_end": int(e),
                "quote": q,
            }
        ],
        "tags": [c.title_ar] if c.title_ar else [],
        "version": version,
    }


async def _run(*, out_path: Path, limit: int, version: str) -> None:
    candidates = await _fetch_candidates(limit=limit)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for c in candidates:
            row = _note_row(c, version=version)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(candidates)} notes -> {out_path}")


def main() -> None:
    load_dotenv_if_present()

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/scholar_notes/notes_v1.jsonl")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--version", default="v1")
    args = p.parse_args()

    asyncio.run(_run(out_path=Path(args.out), limit=int(args.limit), version=str(args.version)))


if __name__ == "__main__":
    main()

