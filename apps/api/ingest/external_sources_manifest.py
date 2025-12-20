"""External corpus manifest (provenance + hashes).

This supports Phase 2 corpus expansion via file-based sources provided by the user.

Hard gates:
- No ingestion without a manifest row.
- SHA256 must match the referenced file bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ExternalSourceManifestRow(BaseModel):
    """
    One external source to ingest.

    Notes:
    - `file_path` is relative to the repo root unless absolute.
    - `sha256` must match the file bytes exactly.
    """

    source_id: str = Field(..., min_length=3, max_length=80)
    title: str = Field(..., min_length=1, max_length=300)
    author: Optional[str] = Field(default=None, max_length=200)
    year: Optional[int] = Field(default=None, ge=1, le=3000)
    license: str = Field(..., min_length=1, max_length=200)

    file_path: str = Field(..., min_length=1, max_length=500)
    sha256: str = Field(..., min_length=64, max_length=64, description="Hex SHA256 of file bytes")

    file_format: str = Field(default="txt", description="txt|md (pdf not supported in repo by default)")
    language: str = Field(default="ar", description="ar|en|mixed")
    weight: float = Field(default=1.0, ge=0.0, le=10.0)

    @field_validator("sha256")
    @classmethod
    def _sha256_hex(cls, v: str) -> str:
        vv = (v or "").strip().lower()
        if len(vv) != 64 or any(ch not in "0123456789abcdef" for ch in vv):
            raise ValueError("sha256 must be 64 hex chars")
        return vv

    @field_validator("file_format")
    @classmethod
    def _fmt(cls, v: str) -> str:
        vv = (v or "").strip().lower()
        if vv not in {"txt", "md"}:
            raise ValueError("file_format must be txt|md")
        return vv


def sha256_file_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_external_manifest(*, manifest_path: str | Path) -> list[ExternalSourceManifestRow]:
    p = Path(manifest_path)
    rows: list[ExternalSourceManifestRow] = []
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        rows.append(ExternalSourceManifestRow.model_validate(obj))
    return rows

