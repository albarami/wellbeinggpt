"""Cross-pillar graph pair extraction for datasets.

We represent edges by a canonical, DB-derivable string ID:
  edge_id = "{from_type}:{from_id}::{rel_type}::{to_type}:{to_id}"

Reason: DB edge UUIDs are not stable across re-ingestion, but the natural key is.
Scorers validate existence by querying the edge table using this natural key.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


def canonical_edge_id(
    *,
    from_type: str,
    from_id: str,
    rel_type: str,
    to_type: str,
    to_id: str,
) -> str:
    return f"{from_type}:{from_id}::{rel_type}::{to_type}:{to_id}"


@dataclass(frozen=True)
class EntityInfo:
    entity_type: str
    entity_id: str
    name_ar: str
    pillar_id: str


def build_entity_index(canonical_json: dict[str, Any]) -> dict[tuple[str, str], EntityInfo]:
    out: dict[tuple[str, str], EntityInfo] = {}

    for p in canonical_json.get("pillars", []) or []:
        pid = str(p.get("id") or "")
        pname = str(p.get("name_ar") or "")
        if pid:
            out[("pillar", pid)] = EntityInfo("pillar", pid, pname, pid)

        for cv in (p.get("core_values") or []):
            cvid = str(cv.get("id") or "")
            cvname = str(cv.get("name_ar") or "")
            if cvid:
                out[("core_value", cvid)] = EntityInfo("core_value", cvid, cvname, pid)

            for sv in (cv.get("sub_values") or []):
                svid = str(sv.get("id") or "")
                svname = str(sv.get("name_ar") or "")
                if svid:
                    out[("sub_value", svid)] = EntityInfo("sub_value", svid, svname, pid)

    return out


def _chunk_refs(chunk: dict[str, Any]) -> list[str]:
    refs = chunk.get("refs") or []
    out: list[str] = []
    if not isinstance(refs, list):
        return out
    for r in refs:
        if not isinstance(r, dict):
            continue
        t = (r.get("type") or "").strip()
        ref = (r.get("ref") or "").strip()
        if t and ref:
            out.append(f"{t}:{ref}")
    return out


def cross_pillar_pairs_from_chunks(
    *,
    entity_index: dict[tuple[str, str], EntityInfo],
    chunks_rows: Iterable[dict[str, Any]],
    max_pairs: int = 400,
) -> list[dict[str, Any]]:
    """Derive cross-pillar pairs from shared refs and same-name.

    Output rows are deterministic and use canonical edge ids.
    """
    # Map ref -> set of entities
    entities_by_ref: dict[str, list[EntityInfo]] = defaultdict(list)

    # Same-name for sub-values
    sub_by_name: dict[str, list[EntityInfo]] = defaultdict(list)

    for ch in chunks_rows:
        et = str(ch.get("entity_type") or "")
        eid = str(ch.get("entity_id") or "")
        info = entity_index.get((et, eid))
        if not info:
            continue

        if info.entity_type == "sub_value" and info.name_ar:
            sub_by_name[info.name_ar].append(info)

        for rr in _chunk_refs(ch):
            entities_by_ref[rr].append(info)

    out: list[dict[str, Any]] = []

    # Prefer SHARES_REF
    for ref_id in sorted(entities_by_ref.keys()):
        ents = entities_by_ref[ref_id]
        # Deterministic unique entities
        uniq: dict[tuple[str, str], EntityInfo] = {}
        for e in ents:
            uniq[(e.entity_type, e.entity_id)] = e
        e_list = list(uniq.values())
        e_list.sort(key=lambda x: (x.pillar_id, x.entity_type, x.entity_id))

        # Cross-pillar pairs only
        for i in range(len(e_list)):
            for j in range(i + 1, len(e_list)):
                a = e_list[i]
                b = e_list[j]
                if a.pillar_id == b.pillar_id:
                    continue

                # Canonical direction
                from_e, to_e = (a, b) if (a.entity_type, a.entity_id) <= (b.entity_type, b.entity_id) else (b, a)
                out.append(
                    {
                        "rel_type": "SHARES_REF",
                        "edge_id": canonical_edge_id(
                            from_type=from_e.entity_type,
                            from_id=from_e.entity_id,
                            rel_type="SHARES_REF",
                            to_type=to_e.entity_type,
                            to_id=to_e.entity_id,
                        ),
                        "e1": from_e,
                        "e2": to_e,
                        "justification": ref_id,
                    }
                )
                if len(out) >= max_pairs:
                    return out

    # Then SAME_NAME (sub_value only)
    for name in sorted(sub_by_name.keys()):
        ents = sub_by_name[name]
        uniq: dict[tuple[str, str], EntityInfo] = {}
        for e in ents:
            uniq[(e.entity_type, e.entity_id)] = e
        e_list = list(uniq.values())
        e_list.sort(key=lambda x: (x.pillar_id, x.entity_id))

        for i in range(len(e_list)):
            for j in range(i + 1, len(e_list)):
                a = e_list[i]
                b = e_list[j]
                if a.pillar_id == b.pillar_id:
                    continue

                from_e, to_e = (a, b) if a.entity_id <= b.entity_id else (b, a)
                out.append(
                    {
                        "rel_type": "SAME_NAME",
                        "edge_id": canonical_edge_id(
                            from_type=from_e.entity_type,
                            from_id=from_e.entity_id,
                            rel_type="SAME_NAME",
                            to_type=to_e.entity_type,
                            to_id=to_e.entity_id,
                        ),
                        "e1": from_e,
                        "e2": to_e,
                        "justification": name,
                    }
                )
                if len(out) >= max_pairs:
                    return out

    return out
