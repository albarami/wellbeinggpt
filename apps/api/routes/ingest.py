"""
Ingestion routes for document processing.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from apps.api.core.database import get_session
from apps.api.ingest.docx_reader import DocxReader
from apps.api.ingest.ocr_augment import augment_document_with_image_ocr
from apps.api.ingest.rule_extractor import RuleExtractor
from apps.api.ingest.validator import validate_extraction, ValidationSeverity
from apps.api.ingest.canonical_json import extraction_to_canonical_json
from apps.api.ingest.pipeline_framework import _expand_evidence_in_canonical
from apps.api.ingest.chunker import Chunker
from apps.api.ingest.loader import load_canonical_json_to_db
from apps.api.ingest.supplemental_ocr import save_supplemental_ocr_text
from apps.api.llm.vision_ocr_azure import VisionOcrClient, VisionOcrConfig

router = APIRouter()


class IngestionRunResponse(BaseModel):
    """Response model for ingestion run."""

    run_id: str
    status: str
    source_file_name: str
    source_doc_id: Optional[str] = None
    source_file_hash: Optional[str] = None
    created_at: datetime
    message: str


class IngestionRunStatus(BaseModel):
    """Status model for ingestion run."""

    run_id: str
    status: str
    entities_extracted: int
    evidence_extracted: int
    validation_errors: list[str]
    created_at: datetime
    completed_at: Optional[datetime] = None


@router.post("/docx", response_model=IngestionRunResponse)
async def ingest_docx(file: UploadFile = File(...)):
    """
    Ingest an Arabic framework document (.docx).

    Extracts:
    - Pillars (ركائز)
    - Core values (القيم الكلية/الأمهات)
    - Sub-values (القيم الجزئية/الأحفاد)
    - Definitions (المفهوم)
    - Evidence (التأصيل) with Quran/Hadith references

    Returns:
        IngestionRunResponse: Run ID and status for tracking.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(
            status_code=400,
            detail="Only .docx files are supported",
        )

    content = await file.read()
    reader = DocxReader()
    parsed = reader.read_bytes(content, file.filename)
    # OCR augmentation for image-based pages inside the DOCX (ingestion-only).
    # This ensures all values/evidence embedded as non-selectable images are included in Postgres/RAG/graph.
    parsed, _ocr_stats = await augment_document_with_image_ocr(parsed, content)

    extractor = RuleExtractor(framework_version="2025-10")
    extracted = extractor.extract(parsed)

    validation = validate_extraction(extracted, strict=True)
    if not validation.is_valid:
        errors = [i.message for i in validation.issues if i.severity == ValidationSeverity.ERROR]
        raise HTTPException(status_code=400, detail={"errors": errors[:20]})

    canonical = extraction_to_canonical_json(extracted)
    canonical = _expand_evidence_in_canonical(canonical)

    # Build chunks JSONL to local derived folder (optional, but used by loader)
    from pathlib import Path
    import os

    derived_dir = Path("data/derived")
    derived_dir.mkdir(parents=True, exist_ok=True)
    base = Path(file.filename).stem
    canonical_path = derived_dir / f"{base}.canonical.json"
    chunks_path = derived_dir / f"{base}.chunks.jsonl"
    canonical.setdefault("meta", {})
    canonical["meta"]["canonical_path"] = str(canonical_path)
    canonical["meta"]["chunks_path"] = str(chunks_path)

    # Save canonical + chunks
    from apps.api.ingest.canonical_json import save_canonical_json
    save_canonical_json(canonical, canonical_path)
    chunks = Chunker().chunk_canonical_json(canonical)
    Chunker().save_chunks_jsonl(chunks, str(chunks_path))

    async with get_session() as session:
        summary = await load_canonical_json_to_db(session, canonical, file.filename)

    return IngestionRunResponse(
        run_id=summary["run_id"],
        status="completed",
        source_file_name=file.filename,
        source_doc_id=summary.get("source_doc_id"),
        source_file_hash=canonical.get("meta", {}).get("source_file_hash"),
        created_at=datetime.utcnow(),
        message=f"Ingestion completed. pillars={summary['pillars']} core_values={summary['core_values']} sub_values={summary['sub_values']} chunks={summary.get('chunks',0)} embeddings={summary.get('embeddings',0)}",
    )


class SupplementalImagesResponse(BaseModel):
    source_file_hash: str
    images_received: int
    images_written: int
    message: str


@router.post("/supplemental-images", response_model=SupplementalImagesResponse)
async def upload_supplemental_images(
    files: List[UploadFile] = File(...),
    source_file_hash: Optional[str] = Form(default=None),
    source_doc_id: Optional[str] = Form(default=None),
):
    """
    Upload supplemental screenshots for OCR ingestion.

    These screenshots are OCR'd and stored (text only) under `data/derived/` (gitignored).
    Subsequent `/ingest/docx` runs will automatically include them.

    Provide either:
    - source_file_hash (preferred): sha256 of the DOCX bytes (same as source_document.file_hash)
    - source_doc_id: UUID of the source_document row (we resolve to file_hash)
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Resolve source_file_hash from DB if source_doc_id is provided
    if (not source_file_hash) and source_doc_id:
        async with get_session() as session:
            from sqlalchemy import text as sqltext

            row = (
                await session.execute(
                    sqltext("SELECT file_hash FROM source_document WHERE id = :id"),
                    {"id": source_doc_id},
                )
            ).fetchone()
            if row and row.file_hash:
                source_file_hash = str(row.file_hash)

    if not source_file_hash:
        raise HTTPException(
            status_code=400,
            detail="Provide source_file_hash or source_doc_id.",
        )

    cfg = VisionOcrConfig.from_env()
    if not cfg.is_configured():
        raise HTTPException(status_code=500, detail="Vision OCR is not configured.")
    client = VisionOcrClient(cfg)

    written = 0
    for f in files:
        img_bytes = await f.read()
        if not img_bytes:
            continue
        res = await client.ocr_image(img_bytes)
        if res.error or not res.text_ar.strip():
            continue
        lines = [ln.strip() for ln in res.text_ar.splitlines() if ln.strip()]
        save_supplemental_ocr_text(
            source_file_hash=source_file_hash,
            image_sha256=res.image_sha256,
            filename=f.filename or "upload.png",
            lines=lines,
            context={},
        )
        written += 1

    return SupplementalImagesResponse(
        source_file_hash=source_file_hash,
        images_received=len(files),
        images_written=written,
        message="Supplemental OCR text stored. Re-run /ingest/docx to incorporate into DB/graph/RAG.",
    )


@router.get("/runs/{run_id}", response_model=IngestionRunStatus)
async def get_ingestion_run(run_id: str):
    """
    Get the status of an ingestion run.

    Args:
        run_id: The unique identifier of the ingestion run.

    Returns:
        IngestionRunStatus: Current status and statistics.
    """
    # Best-effort status lookup from DB
    async with get_session() as session:
        from sqlalchemy import text
        result = await session.execute(
            text(
                """
                SELECT id, status, entities_extracted, evidence_extracted, validation_errors, created_at, completed_at
                FROM ingestion_run
                WHERE id = :id
                """
            ),
            {"id": run_id},
        )
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")

        try:
            errors = row.validation_errors if isinstance(row.validation_errors, list) else []
        except Exception:
            errors = []

        return IngestionRunStatus(
            run_id=str(row.id),
            status=row.status,
            entities_extracted=row.entities_extracted or 0,
            evidence_extracted=row.evidence_extracted or 0,
            validation_errors=errors,
            created_at=row.created_at,
            completed_at=row.completed_at,
        )

