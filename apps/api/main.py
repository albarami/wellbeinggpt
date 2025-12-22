"""
Wellbeing Data Foundation API

A zero-hallucination, evidence-only Arabic wellbeing assistant.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routes import ingest, ask, graph, graph_ui, resolve, ui, ui_runs
from apps.api.llm.gpt5_client_azure import ProviderConfig

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
app.include_router(graph.router, tags=["graph"])
app.include_router(graph_ui.router, tags=["graph"])
app.include_router(resolve.router, tags=["resolver"])
app.include_router(ui.router, tags=["ui"])
app.include_router(ui_runs.router, tags=["ui"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    cfg = ProviderConfig.from_env()
    
    # Edge trace stats
    edge_trace_info = {}
    try:
        from apps.api.graph.edge_candidate_logger import get_edge_trace_stats, EDGE_SCORING_INTENTS
        edge_trace_info = get_edge_trace_stats()
        edge_trace_info["enabled"] = os.getenv("EDGE_TRACE_LOGGING", "false")
        edge_trace_info["scoring_intents"] = list(EDGE_SCORING_INTENTS)
    except Exception as e:
        edge_trace_info = {"error": str(e)}
    
    return {
        "status": "healthy",
        "version": "0.1.0",
        "llm": {
            "provider_type": cfg.provider_type.value,
            "configured": cfg.is_configured(),
            "deployment_name_set": bool(cfg.deployment_name),
            "endpoint_set": bool(cfg.endpoint),
            "api_version": cfg.api_version,
        },
        "edge_trace": edge_trace_info,
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Wellbeing Data Foundation API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }

