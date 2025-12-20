"""Tests for World Model loop detection.

Tests:
- Polarity classification (product of signs)
- Deterministic loop detection
- Relevance scoring
- Summary generation from spans
"""

import pytest

from apps.api.core.world_model.schemas import (
    compute_loop_type,
    DetectedLoop,
    compute_edge_confidence,
    get_default_polarity,
    RELATION_POLARITY_DEFAULTS,
)
from apps.api.core.world_model.loop_reasoner import (
    compute_loop_relevance_score,
    retrieve_relevant_loops,
)


class TestPolarityClassification:
    """Tests for loop polarity classification via product of signs."""
    
    def test_all_positive_is_reinforcing(self):
        """All positive edges → reinforcing loop."""
        assert compute_loop_type([1, 1, 1]) == "reinforcing"
        assert compute_loop_type([1, 1]) == "reinforcing"
        assert compute_loop_type([1]) == "reinforcing"
    
    def test_even_negatives_is_reinforcing(self):
        """Even number of negative edges → reinforcing loop."""
        assert compute_loop_type([1, -1, -1]) == "reinforcing"
        assert compute_loop_type([-1, -1]) == "reinforcing"
        assert compute_loop_type([1, 1, -1, -1]) == "reinforcing"
        assert compute_loop_type([-1, -1, -1, -1]) == "reinforcing"
    
    def test_odd_negatives_is_balancing(self):
        """Odd number of negative edges → balancing loop."""
        assert compute_loop_type([1, 1, -1]) == "balancing"
        assert compute_loop_type([-1]) == "balancing"
        assert compute_loop_type([1, -1, -1, -1]) == "balancing"
        assert compute_loop_type([1, -1]) == "balancing"
    
    def test_empty_list_defaults_to_reinforcing(self):
        """Empty polarity list defaults to reinforcing."""
        assert compute_loop_type([]) == "reinforcing"
    
    def test_mixed_polarities(self):
        """Complex mixed polarity cases."""
        # Two positives, one negative = balancing
        assert compute_loop_type([1, 1, -1]) == "balancing"
        # Three positives, two negatives = reinforcing
        assert compute_loop_type([1, 1, 1, -1, -1]) == "reinforcing"
        # Four edges with three negatives = balancing
        assert compute_loop_type([1, -1, -1, -1]) == "balancing"


class TestDefaultPolarities:
    """Tests for relation type default polarities."""
    
    def test_positive_relations(self):
        """ENABLES, REINFORCES, COMPLEMENTS, RESOLVES_WITH, CONDITIONAL_ON are +1."""
        assert get_default_polarity("ENABLES") == 1
        assert get_default_polarity("REINFORCES") == 1
        assert get_default_polarity("COMPLEMENTS") == 1
        assert get_default_polarity("RESOLVES_WITH") == 1
        assert get_default_polarity("CONDITIONAL_ON") == 1
    
    def test_negative_relations(self):
        """INHIBITS, TENSION_WITH are -1."""
        assert get_default_polarity("INHIBITS") == -1
        assert get_default_polarity("TENSION_WITH") == -1
    
    def test_unknown_defaults_to_positive(self):
        """Unknown relation types default to +1."""
        assert get_default_polarity("UNKNOWN") == 1
        assert get_default_polarity("") == 1
    
    def test_case_insensitive(self):
        """Relation type lookup is case-insensitive."""
        assert get_default_polarity("enables") == 1
        assert get_default_polarity("ENABLES") == 1
        assert get_default_polarity("inhibits") == -1


class TestEdgeConfidence:
    """Tests for evidence-based confidence computation."""
    
    def test_single_span_single_chunk(self):
        """Single span from single chunk."""
        confidence = compute_edge_confidence(
            span_count=1,
            chunk_diversity=1,
            is_direct_quote=True,
        )
        # base (0.1 + 0.1) + diversity (0.05) + direct_quote (0.1) = 0.35
        assert 0.3 <= confidence <= 0.4
    
    def test_multiple_spans_increase_confidence(self):
        """More spans → higher confidence."""
        c1 = compute_edge_confidence(span_count=1, chunk_diversity=1, is_direct_quote=True)
        c2 = compute_edge_confidence(span_count=3, chunk_diversity=1, is_direct_quote=True)
        c3 = compute_edge_confidence(span_count=5, chunk_diversity=1, is_direct_quote=True)
        
        assert c1 < c2 < c3
    
    def test_chunk_diversity_increases_confidence(self):
        """Spans from different chunks → higher confidence."""
        c1 = compute_edge_confidence(span_count=3, chunk_diversity=1, is_direct_quote=True)
        c2 = compute_edge_confidence(span_count=3, chunk_diversity=3, is_direct_quote=True)
        
        assert c1 < c2
    
    def test_direct_quote_bonus(self):
        """Direct quote gets bonus over multi-span entailment."""
        c_direct = compute_edge_confidence(span_count=1, chunk_diversity=1, is_direct_quote=True)
        c_multi = compute_edge_confidence(span_count=1, chunk_diversity=1, is_direct_quote=False)
        
        assert c_direct > c_multi
    
    def test_confidence_in_valid_range(self):
        """Confidence is always in [0.1, 0.95] range."""
        # Minimum case
        c_min = compute_edge_confidence(span_count=0, chunk_diversity=0, is_direct_quote=False)
        assert c_min >= 0.1
        
        # Maximum case
        c_max = compute_edge_confidence(span_count=10, chunk_diversity=10, is_direct_quote=True)
        assert c_max <= 0.95


class TestLoopRelevanceScoring:
    """Tests for loop relevance scoring."""
    
    def test_matching_pillar_increases_score(self):
        """Loops with matching pillars get higher score."""
        loop = DetectedLoop(
            loop_id="test-1",
            loop_type="reinforcing",
            edge_ids=["e1", "e2"],
            nodes=["pillar:P001", "pillar:P002"],
            node_labels_ar=["الروحية", "العاطفية"],
            polarities=[1, 1],
            evidence_spans=[{"chunk_id": "c1", "quote": "test"}],
        )
        
        # With matching pillars
        score_match = compute_loop_relevance_score(
            loop,
            detected_entities=[],
            detected_pillars=["P001", "P002"],
        )
        
        # Without matching pillars
        score_no_match = compute_loop_relevance_score(
            loop,
            detected_entities=[],
            detected_pillars=["P004", "P005"],
        )
        
        assert score_match > score_no_match
    
    def test_evidence_bonus(self):
        """Loops with more evidence get higher score."""
        loop_no_evidence = DetectedLoop(
            loop_id="test-1",
            loop_type="reinforcing",
            edge_ids=["e1"],
            nodes=["pillar:P001"],
            node_labels_ar=["الروحية"],
            polarities=[1],
            evidence_spans=[],
        )
        
        loop_with_evidence = DetectedLoop(
            loop_id="test-2",
            loop_type="reinforcing",
            edge_ids=["e1"],
            nodes=["pillar:P001"],
            node_labels_ar=["الروحية"],
            polarities=[1],
            evidence_spans=[
                {"chunk_id": "c1", "quote": "evidence 1"},
                {"chunk_id": "c2", "quote": "evidence 2"},
            ],
        )
        
        score_no_ev = compute_loop_relevance_score(
            loop_no_evidence,
            detected_entities=[],
            detected_pillars=["P001"],
        )
        
        score_with_ev = compute_loop_relevance_score(
            loop_with_evidence,
            detected_entities=[],
            detected_pillars=["P001"],
        )
        
        assert score_with_ev >= score_no_ev
    
    def test_retrieve_relevant_loops_ordering(self):
        """retrieve_relevant_loops returns loops ordered by relevance."""
        loops = [
            DetectedLoop(
                loop_id="low",
                loop_type="reinforcing",
                edge_ids=["e1"],
                nodes=["pillar:P004"],
                node_labels_ar=["البدنية"],
                polarities=[1],
                evidence_spans=[],
            ),
            DetectedLoop(
                loop_id="high",
                loop_type="reinforcing",
                edge_ids=["e1"],
                nodes=["pillar:P001", "pillar:P002"],
                node_labels_ar=["الروحية", "العاطفية"],
                polarities=[1, 1],
                evidence_spans=[{"chunk_id": "c1", "quote": "test"}],
            ),
        ]
        
        relevant = retrieve_relevant_loops(
            loops,
            detected_entities=[],
            detected_pillars=["P001", "P002"],
            top_k=2,
        )
        
        # High relevance loop should come first
        assert relevant[0].loop_id == "high"


class TestLoopSummaryGeneration:
    """Tests for on-the-fly loop summary generation."""
    
    def test_summary_from_spans(self):
        """Summary is generated from evidence spans."""
        loop = DetectedLoop(
            loop_id="test",
            loop_type="reinforcing",
            edge_ids=["e1"],
            nodes=["pillar:P001"],
            node_labels_ar=["الروحية"],
            polarities=[1],
            evidence_spans=[{"quote": "هذا دليل من النص"}],
        )
        
        summary = loop.generate_summary_ar()
        assert "هذا دليل" in summary or summary != "غير منصوص عليه في الإطار"
    
    def test_no_spans_returns_placeholder(self):
        """Empty spans returns placeholder text."""
        loop = DetectedLoop(
            loop_id="test",
            loop_type="reinforcing",
            edge_ids=["e1"],
            nodes=["pillar:P001"],
            node_labels_ar=["الروحية"],
            polarities=[1],
            evidence_spans=[],
        )
        
        summary = loop.generate_summary_ar()
        assert summary == "غير منصوص عليه في الإطار"
    
    def test_long_quotes_truncated(self):
        """Long quotes are truncated in summary."""
        long_quote = "أ" * 200  # 200 characters
        loop = DetectedLoop(
            loop_id="test",
            loop_type="reinforcing",
            edge_ids=["e1"],
            nodes=["pillar:P001"],
            node_labels_ar=["الروحية"],
            polarities=[1],
            evidence_spans=[{"quote": long_quote}],
        )
        
        summary = loop.generate_summary_ar()
        assert len(summary) < 200 or "..." in summary
