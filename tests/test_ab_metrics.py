"""Tests for A/B evaluation metrics.

Tests:
- AbSummary computation from sample data
- Delta calculations with correct attribution
- Style metrics (quote count, paragraph count, reasoning leak)
"""

import pytest

from eval.scoring.ab_metrics import (
    AbSummary,
    AbDelta,
    AbAttributionAnalysis,
    _count_quotes,
    _count_paragraphs,
    _has_reasoning_block_leak,
    _distinct_pillars_from_edges,
    compute_mode_summary,
    compute_delta,
    compute_all_deltas,
    compute_attribution_analysis,
)


class TestStyleMetrics:
    """Test style metric helper functions."""

    def test_count_quotes_guillemets(self):
        """Test counting «» quotes."""
        text = "قال النبي «من غشنا فليس منا» وقال «الدين النصيحة»"
        assert _count_quotes(text) == 2

    def test_count_quotes_double(self):
        """Test counting "" quotes."""
        text = 'قال "اتقوا الله" وقال "أوفوا بالعقود"'
        assert _count_quotes(text) == 2

    def test_count_quotes_none(self):
        """Test text with no quotes."""
        text = "هذا نص بدون اقتباسات مباشرة"
        assert _count_quotes(text) == 0

    def test_count_paragraphs_double_newline(self):
        """Test counting paragraphs separated by double newlines."""
        text = "الفقرة الأولى\n\nالفقرة الثانية\n\nالفقرة الثالثة"
        assert _count_paragraphs(text) >= 3

    def test_count_paragraphs_empty(self):
        """Test empty text returns 0 paragraphs."""
        assert _count_paragraphs("") == 0
        assert _count_paragraphs("   ") == 0

    def test_has_reasoning_block_leak_true(self):
        """Test detecting reasoning block leak."""
        text = "الإجابة هي [[MUHASIBI_REASONING_START]] بعض التفكير [[MUHASIBI_REASONING_END]]"
        assert _has_reasoning_block_leak(text) is True

    def test_has_reasoning_block_leak_false(self):
        """Test clean text without leak."""
        text = "هذا نص نظيف بدون أي تسرب للتفكير الداخلي"
        assert _has_reasoning_block_leak(text) is False


class TestPillarCounting:
    """Test pillar extraction from edges."""

    def test_distinct_pillars_from_edges(self):
        """Test counting distinct pillars."""
        edges = [
            {"from_node": "pillar:P001", "to_node": "pillar:P002"},
            {"from_node": "pillar:P002", "to_node": "pillar:P003"},
            {"from_node": "core_value:V001", "to_node": "pillar:P001"},
        ]
        assert _distinct_pillars_from_edges(edges) == 3  # P001, P002, P003

    def test_distinct_pillars_empty(self):
        """Test empty edges list."""
        assert _distinct_pillars_from_edges([]) == 0

    def test_distinct_pillars_no_pillars(self):
        """Test edges without pillar nodes."""
        edges = [
            {"from_node": "core_value:V001", "to_node": "sub_value:S001"},
        ]
        assert _distinct_pillars_from_edges(edges) == 0


class TestAbSummaryComputation:
    """Test AbSummary computation from rows."""

    def test_compute_mode_summary_empty(self):
        """Test summary of empty rows."""
        summary = compute_mode_summary("TEST_MODE", [])
        assert summary.mode == "TEST_MODE"
        assert summary.n == 0
        assert summary.mean_used_edges == 0.0

    def test_compute_mode_summary_basic(self):
        """Test basic summary computation."""
        rows = [
            {
                "latency_ms": 1000,
                "answer_ar": "الإجابة الأولى",
                "graph_trace": {
                    "used_edges": [
                        {"edge_id": "e1", "from_node": "pillar:P001", "to_node": "pillar:P002"},
                    ],
                    "argument_chains": [{"edge_id": "e1"}],
                },
                "debug": {"contract_outcome": "PASS_FULL"},
                "claims": [],
                "citations": [],
            },
            {
                "latency_ms": 2000,
                "answer_ar": "الإجابة الثانية «اقتباس»",
                "graph_trace": {
                    "used_edges": [
                        {"edge_id": "e2", "from_node": "pillar:P001", "to_node": "pillar:P003"},
                        {"edge_id": "e3", "from_node": "pillar:P002", "to_node": "pillar:P003"},
                    ],
                    "argument_chains": [],
                },
                "debug": {"contract_outcome": "PASS_PARTIAL"},
                "claims": [],
                "citations": [],
            },
        ]
        
        summary = compute_mode_summary("RAG_ONLY", rows)
        
        assert summary.mode == "RAG_ONLY"
        assert summary.n == 2
        assert summary.mean_used_edges == 1.5  # (1 + 2) / 2
        assert summary.mean_argument_chains == 0.5  # (1 + 0) / 2
        assert summary.contract_pass_full == 0.5  # 1/2
        assert summary.contract_pass_partial == 0.5  # 1/2
        assert summary.p50_latency_ms == 1000.0
        assert summary.reasoning_block_leak_rate == 0.0


class TestDeltaComputation:
    """Test delta computation between modes."""

    def test_compute_delta_positive(self):
        """Test positive delta computation."""
        baseline = AbSummary(mode="RAG_ONLY", mean_used_edges=3.0, mean_latency_ms=1000)
        full = AbSummary(mode="FULL_SYSTEM", mean_used_edges=10.0, mean_latency_ms=2500)
        
        delta = compute_delta("mean_used_edges", baseline, full)
        
        assert delta.metric == "mean_used_edges"
        assert delta.baseline_value == 3.0
        assert delta.full_value == 10.0
        assert delta.delta == 7.0
        assert delta.delta_pct == pytest.approx(233.33, rel=0.01)

    def test_compute_delta_negative(self):
        """Test negative delta (improvement = reduction)."""
        baseline = AbSummary(mode="RAG_ONLY", unsupported_must_cite_rate=0.15, mean_latency_ms=1000)
        full = AbSummary(mode="FULL_SYSTEM", unsupported_must_cite_rate=0.0, mean_latency_ms=2500)
        
        delta = compute_delta("unsupported_must_cite_rate", baseline, full)
        
        assert delta.delta == -0.15
        assert delta.delta_pct == -100.0

    def test_compute_delta_per_ms(self):
        """Test value-add per millisecond calculation."""
        baseline = AbSummary(mode="RAG_ONLY", contract_pass_full=0.5, mean_latency_ms=1000)
        full = AbSummary(mode="FULL_SYSTEM", contract_pass_full=0.95, mean_latency_ms=2500)
        
        delta = compute_delta("contract_pass_full", baseline, full)
        
        # (0.95 - 0.5) / (2500 - 1000) = 0.45 / 1500 = 0.0003
        assert delta.delta_per_ms == pytest.approx(0.0003, rel=0.01)


class TestAttributionAnalysis:
    """Test attribution analysis for value-add decomposition."""

    def test_compute_attribution_analysis(self):
        """Test full attribution analysis."""
        summaries = {
            "RAG_ONLY": AbSummary(
                mode="RAG_ONLY",
                unsupported_must_cite_rate=0.15,
                contract_pass_full=0.55,
                mean_used_edges=0.0,
                quarantined_cites_blocked=0,
                mean_latency_ms=800,
            ),
            "RAG_ONLY_INTEGRITY": AbSummary(
                mode="RAG_ONLY_INTEGRITY",
                unsupported_must_cite_rate=0.15,
                contract_pass_full=0.55,
                mean_used_edges=0.0,
                quarantined_cites_blocked=2,
                mean_latency_ms=850,
            ),
            "RAG_PLUS_GRAPH": AbSummary(
                mode="RAG_PLUS_GRAPH",
                unsupported_must_cite_rate=0.12,
                contract_pass_full=0.70,
                mean_used_edges=3.2,
                quarantined_cites_blocked=0,
                mean_latency_ms=1200,
            ),
            "RAG_PLUS_GRAPH_INTEGRITY": AbSummary(
                mode="RAG_PLUS_GRAPH_INTEGRITY",
                unsupported_must_cite_rate=0.12,
                contract_pass_full=0.72,
                mean_used_edges=3.2,
                quarantined_cites_blocked=2,
                mean_latency_ms=1250,
            ),
            "FULL_SYSTEM": AbSummary(
                mode="FULL_SYSTEM",
                unsupported_must_cite_rate=0.00,
                contract_pass_full=0.95,
                mean_used_edges=10.5,
                quarantined_cites_blocked=2,
                mean_latency_ms=2500,
            ),
        }
        
        analysis = compute_attribution_analysis(summaries)
        
        # Integrity effect: quarantine goes from 0 to 2
        assert "quarantined_cites_blocked" in analysis.integrity_effect
        assert analysis.integrity_effect["quarantined_cites_blocked"].delta == 2
        
        # Graph effect: used_edges goes from 0 to 3.2
        assert "mean_used_edges" in analysis.graph_effect
        assert analysis.graph_effect["mean_used_edges"].delta == 3.2
        
        # Muḥāsibī effect: contract pass goes from 0.72 to 0.95
        assert "contract_pass_full" in analysis.muhasibi_effect
        assert analysis.muhasibi_effect["contract_pass_full"].delta == pytest.approx(0.23, rel=0.01)
        
        # Total effect: unsupported drops to 0
        assert "unsupported_must_cite_rate" in analysis.total_effect
        assert analysis.total_effect["unsupported_must_cite_rate"].delta == pytest.approx(-0.15, rel=0.01)
