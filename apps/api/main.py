"""
Wellbeing Data Foundation API

A zero-hallucination, evidence-only Arabic wellbeing assistant.
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routes import ingest, ask

# Load local .env (does not override real env vars by default)
load_dotenv()

app = FastAPI(
    title="Wellbeing Data Foundation API",
    description="Evidence-only Arabic wellbeing assistant with Muḥāsibī reasoning middleware",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
app.include_router(ask.router, tags=["query"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Wellbeing Data Foundation API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }

