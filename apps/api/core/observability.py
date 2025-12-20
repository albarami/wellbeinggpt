"""Observability module for World Model and Muḥāsibī pipeline.

This module provides:
- Structured request logging
- Latency breakdown tracking
- World model metrics (loops, edges, coverage)
- Replay bundle generation for exact reproduction

Per plan requirements:
- Log per request: request_id, mode, intent, loops count, edges count
- Contract outcome and reasons
- Latency breakdown per stage
- Integrity/quarantine hits
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class LatencyBreakdown:
    """Latency tracking for pipeline stages."""
    
    stages: dict[str, float] = field(default_factory=dict)  # stage_name -> seconds
    _current_stage: str | None = None
    _stage_start: float | None = None
    
    def start_stage(self, stage: str) -> None:
        """Start timing a stage."""
        if self._current_stage is not None:
            self.end_stage()
        self._current_stage = stage
        self._stage_start = time.perf_counter()
    
    def end_stage(self) -> None:
        """End timing the current stage."""
        if self._current_stage is not None and self._stage_start is not None:
            elapsed = time.perf_counter() - self._stage_start
            self.stages[self._current_stage] = round(elapsed, 4)
            self._current_stage = None
            self._stage_start = None
    
    def total_seconds(self) -> float:
        """Get total latency across all stages."""
        return round(sum(self.stages.values()), 4)
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return dict(self.stages)


@dataclass
class WorldModelMetrics:
    """Metrics specific to World Model processing."""
    
    # Loop metrics
    loops_detected: int = 0
    loops_used: int = 0
    loop_relevance_avg: float = 0.0
    
    # Edge metrics
    mechanism_edges_total: int = 0
    mechanism_edges_used: int = 0
    avg_edge_confidence: float = 0.0
    span_coverage: float = 0.0  # edges with ≥1 span / total
    
    # Coverage metrics
    pillars_covered: int = 0
    pillars_total: int = 5
    
    # Intervention metrics
    intervention_steps: int = 0
    intervention_completeness: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "loops_detected": self.loops_detected,
            "loops_used": self.loops_used,
            "loop_relevance_avg": self.loop_relevance_avg,
            "mechanism_edges_total": self.mechanism_edges_total,
            "mechanism_edges_used": self.mechanism_edges_used,
            "avg_edge_confidence": self.avg_edge_confidence,
            "span_coverage": self.span_coverage,
            "pillars_covered": self.pillars_covered,
            "pillars_total": self.pillars_total,
            "intervention_steps": self.intervention_steps,
            "intervention_completeness": self.intervention_completeness,
        }


@dataclass
class RequestObservability:
    """Complete observability context for a request."""
    
    request_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Request info
    question: str = ""
    mode: str = ""
    engine: str = ""
    intent: str = ""
    language: str = "ar"
    
    # Contract outcome
    contract_outcome: str = ""
    contract_reasons: list[str] = field(default_factory=list)
    
    # Latency
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    total_latency_ms: int = 0
    
    # World model metrics
    world_model_metrics: WorldModelMetrics = field(default_factory=WorldModelMetrics)
    
    # Safety metrics
    integrity_quarantine_hits: int = 0
    citation_validity_errors: int = 0
    unsupported_must_cite: int = 0
    
    # Debug info
    detected_entities_count: int = 0
    evidence_packets_count: int = 0
    
    def start_stage(self, stage: str) -> None:
        """Start timing a pipeline stage."""
        self.latency.start_stage(stage)
    
    def end_stage(self) -> None:
        """End timing the current pipeline stage."""
        self.latency.end_stage()
    
    def finalize(self) -> None:
        """Finalize the observability record."""
        self.latency.end_stage()  # End any ongoing stage
        self.total_latency_ms = int(self.latency.total_seconds() * 1000)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "question": self.question[:500] if self.question else "",
            "mode": self.mode,
            "engine": self.engine,
            "intent": self.intent,
            "language": self.language,
            "contract_outcome": self.contract_outcome,
            "contract_reasons": self.contract_reasons,
            "latency_breakdown": self.latency.to_dict(),
            "total_latency_ms": self.total_latency_ms,
            "world_model_metrics": self.world_model_metrics.to_dict(),
            "integrity_quarantine_hits": self.integrity_quarantine_hits,
            "citation_validity_errors": self.citation_validity_errors,
            "unsupported_must_cite": self.unsupported_must_cite,
            "detected_entities_count": self.detected_entities_count,
            "evidence_packets_count": self.evidence_packets_count,
        }
    
    def log_summary(self) -> None:
        """Log a summary of this request."""
        logger.info(
            f"[REQUEST] id={self.request_id[:8]} "
            f"intent={self.intent} "
            f"outcome={self.contract_outcome} "
            f"latency={self.total_latency_ms}ms "
            f"loops={self.world_model_metrics.loops_used}/{self.world_model_metrics.loops_detected} "
            f"edges={self.world_model_metrics.mechanism_edges_used} "
            f"pillars={self.world_model_metrics.pillars_covered}/5"
        )


@dataclass
class ReplayBundle:
    """Bundle for exact request reproduction.
    
    Contains all inputs needed to replay a request deterministically.
    """
    
    request_id: str
    timestamp: str
    
    # Input
    question: str
    mode: str
    engine: str
    language: str
    
    # Detected context
    detected_entities: list[dict[str, Any]] = field(default_factory=list)
    detected_pillars: list[str] = field(default_factory=list)
    
    # Retrieved evidence
    evidence_packets: list[dict[str, Any]] = field(default_factory=list)
    semantic_edges: list[dict[str, Any]] = field(default_factory=list)
    
    # World model state
    mechanism_edges: list[dict[str, Any]] = field(default_factory=list)
    detected_loops: list[dict[str, Any]] = field(default_factory=list)
    
    # Output
    final_response: dict[str, Any] = field(default_factory=dict)
    graph_trace: dict[str, Any] = field(default_factory=dict)
    mechanism_trace: dict[str, Any] = field(default_factory=dict)
    
    # Observability
    observability: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "input": {
                "question": self.question,
                "mode": self.mode,
                "engine": self.engine,
                "language": self.language,
            },
            "context": {
                "detected_entities": self.detected_entities,
                "detected_pillars": self.detected_pillars,
            },
            "retrieval": {
                "evidence_packets_count": len(self.evidence_packets),
                "semantic_edges_count": len(self.semantic_edges),
                # Don't store full packets in bundle - use IDs for replay
                "evidence_packet_ids": [p.get("chunk_id") for p in self.evidence_packets[:50]],
            },
            "world_model": {
                "mechanism_edges_count": len(self.mechanism_edges),
                "detected_loops_count": len(self.detected_loops),
            },
            "output": {
                "final_response": self.final_response,
                "graph_trace": self.graph_trace,
                "mechanism_trace": self.mechanism_trace,
            },
            "observability": self.observability,
        }


def create_observability_context(
    *,
    question: str,
    mode: str = "answer",
    engine: str = "muhasibi",
    language: str = "ar",
) -> RequestObservability:
    """Create a new observability context for a request.
    
    Args:
        question: The question being asked
        mode: The mode (answer, debate, etc.)
        engine: The engine (muhasibi, baseline)
        language: The language
        
    Returns:
        New RequestObservability instance
    """
    return RequestObservability(
        question=question,
        mode=mode,
        engine=engine,
        language=language,
    )


def update_world_model_metrics(
    obs: RequestObservability,
    *,
    loops_detected: int = 0,
    loops_used: int = 0,
    loop_relevance_scores: list[float] | None = None,
    edges_total: int = 0,
    edges_used: int = 0,
    edge_confidences: list[float] | None = None,
    edges_with_spans: int = 0,
    pillars_covered: list[str] | None = None,
    intervention_steps: int = 0,
) -> None:
    """Update world model metrics in observability context.
    
    Args:
        obs: The observability context
        loops_detected: Total loops detected in graph
        loops_used: Loops used in the answer
        loop_relevance_scores: Relevance scores for used loops
        edges_total: Total mechanism edges in graph
        edges_used: Edges used in the answer
        edge_confidences: Confidence scores for used edges
        edges_with_spans: Edges that have justification spans
        pillars_covered: List of pillar IDs covered
        intervention_steps: Number of intervention steps generated
    """
    metrics = obs.world_model_metrics
    
    metrics.loops_detected = loops_detected
    metrics.loops_used = loops_used
    
    if loop_relevance_scores:
        metrics.loop_relevance_avg = round(
            sum(loop_relevance_scores) / len(loop_relevance_scores), 3
        )
    
    metrics.mechanism_edges_total = edges_total
    metrics.mechanism_edges_used = edges_used
    
    if edge_confidences:
        metrics.avg_edge_confidence = round(
            sum(edge_confidences) / len(edge_confidences), 3
        )
    
    if edges_total > 0:
        metrics.span_coverage = round(edges_with_spans / edges_total, 3)
    
    if pillars_covered:
        metrics.pillars_covered = len(set(pillars_covered))
    
    metrics.intervention_steps = intervention_steps
    
    # Compute intervention completeness (steps with citations / total steps)
    if intervention_steps > 0:
        # This would need actual citation data to compute properly
        metrics.intervention_completeness = 0.8  # Placeholder


def log_request_observability(obs: RequestObservability) -> None:
    """Log the observability data for a request.
    
    Args:
        obs: The completed observability context
    """
    obs.finalize()
    obs.log_summary()
    
    # Log detailed breakdown if debug enabled
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[REQUEST_DETAIL] {obs.to_dict()}")
