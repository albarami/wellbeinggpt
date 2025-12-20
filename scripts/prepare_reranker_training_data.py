"""Prepare reranker training pairs from eval outputs (grounded).

This script extracts (query, doc_text) pairs with labels using FULL_SYSTEM outputs:
- Positives: cited chunks
- Negatives: retrieved-but-not-cited chunks

Usage:
  python -m scripts.prepare_reranker_training_data ^
    --outputs eval/output/wellbeing__v1__...__FULL_SYSTEM.jsonl ^
    --out data/reranker/train_pairs.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import text

from apps.api.core.database import get_session


async def _fetch_chunk_texts(session, chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    rows = (
        await session.execute(
            text(
                """
                SELECT chunk_id, text_ar
                FROM chunk
                WHERE chunk_id = ANY(:ids)
                """
            ),
            {"ids": chunk_ids},
        )
    ).fetchall()
    return {str(r.chunk_id): str(r.text_ar or "") for r in rows}


async def _run(*, outputs_path: Path, out_path: Path, max_rows: int, neg_per_pos: int) -> None:
    raw_rows = [json.loads(l) for l in outputs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    raw_rows = raw_rows[: max(0, int(max_rows))]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    async with get_session() as session:
        for r in raw_rows:
            if bool(r.get("abstained")):
                continue
            q = str(r.get("question") or "").strip()
            if not q:
                continue
            cits = r.get("citations") or []
            pos_ids = [str(c.get("source_id") or "") for c in cits if str(c.get("source_id") or "")]
            pos_ids = list(dict.fromkeys(pos_ids))[:5]
            if not pos_ids:
                continue

            retrieved = r.get("retrieval_trace", {}).get("top_k_chunks") or []
            neg_ids = [str(x.get("chunk_id") or "") for x in retrieved if str(x.get("chunk_id") or "")]
            neg_ids = [cid for cid in neg_ids if cid and cid not in set(pos_ids)]
            neg_ids = neg_ids[: max(0, int(neg_per_pos) * len(pos_ids) + 5)]

            texts = await _fetch_chunk_texts(session, list(dict.fromkeys(pos_ids + neg_ids)))
            for pid in pos_ids:
                ptxt = (texts.get(pid) or "").strip()
                if not ptxt:
                    continue
                # One positive pair
                row_out = {"query": q, "chunk_id": pid, "text_ar": ptxt, "label": 1}
                out_path.write_text("", encoding="utf-8") if n_written == 0 else None
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                n_written += 1

                # Negatives for this positive
                for nid in neg_ids[: int(neg_per_pos)]:
                    ntxt = (texts.get(nid) or "").strip()
                    if not ntxt:
                        continue
                    row_out = {"query": q, "chunk_id": nid, "text_ar": ntxt, "label": 0}
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                    n_written += 1

    print(f"Wrote {n_written} pairs -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-rows", type=int, default=5000)
    p.add_argument("--neg-per-pos", type=int, default=2)
    args = p.parse_args()
    asyncio.run(
        _run(
            outputs_path=Path(args.outputs),
            out_path=Path(args.out),
            max_rows=int(args.max_rows),
            neg_per_pos=int(args.neg_per_pos),
        )
    )


if __name__ == "__main__":
    main()

