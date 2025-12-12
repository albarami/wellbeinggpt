"""
Ingestion routes for document processing.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime

router = APIRouter()


class IngestionRunResponse(BaseModel):
    """Response model for ingestion run."""

    run_id: str
    status: str
    source_file_name: str
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

    run_id = str(uuid.uuid4())

    # TODO: Implement actual ingestion pipeline in Phase 1
    # This is a placeholder that will be replaced

    return IngestionRunResponse(
        run_id=run_id,
        status="pending",
        source_file_name=file.filename,
        created_at=datetime.utcnow(),
        message="Ingestion queued. Implementation pending Phase 1.",
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
    # TODO: Implement actual status retrieval in Phase 1

    return IngestionRunStatus(
        run_id=run_id,
        status="pending",
        entities_extracted=0,
        evidence_extracted=0,
        validation_errors=[],
        created_at=datetime.utcnow(),
    )

