"""Tests for breakthrough features.

Tests:
- Candidate ranker determinism and scoring
- Candidate generator determinism
- Critic loop fixable-only triggers
"""

import pytest

from apps.api.core.breakthrough.candidate_ranker import (
    Candidate,
    score_candidate,
    pick_best,
    rank_candidates,
    WEIGHT_PILLARS,
    WEIGHT_EVIDENCE_SPANS,
    WEIGHT_BOUNDARY,
    WEIGHT_INFERENCE_PENALTY,
)
from apps.api.core.breakthrough.candidate_generator import (
    generate_candidates_sync,
    _generate_candidate_all_edges,
    _generate_candidate_by_pillar_diversity,
    _generate_candidate_by_relation_type,
)
from apps.api.core.breakthrough.critic_loop import (
    FIXABLE_FAILURES,
    NON_FIXABLE_FAILURES,
    should_trigger_critic,
    critique_draft,
    _classify_contract_reasons,
)


class TestCandidateRanker:
    """Test candidate ranking determinism and scoring."""

    def test_score_candidate_basic(self):
        """Test basic scoring formula."""
        candidate = Candidate(
            name="test_cand",
            used_edges=[
                {
                    "edge_id": "e1",
                    "from_node": "pillar:P001",
                    "to_node": "pillar:P002",
                    "justification_spans": [{"chunk_id": "c1", "quote": "test"}],
                },
            ],
            distinct_pillars=2,
            evidence_span_count=1,
            boundary_hits=1,
            inference_penalty=0.0,
        )
        
        scored = score_candidate(candidate)
        
        # score = 3.0 * 2 + 0.3 * 1 + 1.5 * 1 - 2.0 * 0.0 = 6 + 0.3 + 1.5 = 7.8
        expected = (WEIGHT_PILLARS * 2 + WEIGHT_EVIDENCE_SPANS * 1 + 
                   WEIGHT_BOUNDARY * 1 - WEIGHT_INFERENCE_PENALTY * 0.0)
        assert scored.score == expected

    def test_score_candidate_with_penalty(self):
        """Test scoring with inference penalty."""
        candidate = Candidate(
            name="test_cand",
            used_edges=[
                {
                    "edge_id": "e1",
                    "from_node": "pillar:P001",
                    "to_node": "pillar:P002",
                    "justification_spans": [
                        {"chunk_id": "c1", "quote": "test1"},
                        {"chunk_id": "c2", "quote": "test2"},
                    ],
                },
            ],
            distinct_pillars=2,
            evidence_span_count=2,
            boundary_hits=0,
        )
        
        scored = score_candidate(candidate)
        
        # Penalty for 2 spans = 0.2
        assert scored.inference_penalty == 0.2
        expected = (WEIGHT_PILLARS * 2 + WEIGHT_EVIDENCE_SPANS * 2 + 
                   WEIGHT_BOUNDARY * 0 - WEIGHT_INFERENCE_PENALTY * 0.2)
        assert scored.score == expected

    def test_pick_best_single(self):
        """Test picking best with single candidate."""
        candidates = [
            Candidate(name="only_one", distinct_pillars=1, evidence_span_count=1),
        ]
        
        best = pick_best(candidates)
        
        assert best is not None
        assert best.name == "only_one"

    def test_pick_best_deterministic_tie_break(self):
        """Test deterministic tie-breaking by name."""
        # Create two candidates with identical scores
        candidates = [
            Candidate(name="b_second", distinct_pillars=2, evidence_span_count=1),
            Candidate(name="a_first", distinct_pillars=2, evidence_span_count=1),
        ]
        
        best = pick_best(candidates)
        
        # Should pick "a_first" due to lexicographic tie-break
        assert best.name == "a_first"

    def test_pick_best_by_score(self):
        """Test picking best by score."""
        candidates = [
            Candidate(name="low", distinct_pillars=1, evidence_span_count=1),
            Candidate(name="high", distinct_pillars=5, evidence_span_count=10),
        ]
        
        best = pick_best(candidates)
        
        assert best.name == "high"

    def test_pick_best_empty(self):
        """Test picking from empty list returns None."""
        assert pick_best([]) is None

    def test_rank_candidates_deterministic(self):
        """Test that ranking is deterministic."""
        candidates = [
            Candidate(name="c", distinct_pillars=2, evidence_span_count=1),
            Candidate(name="a", distinct_pillars=3, evidence_span_count=2),
            Candidate(name="b", distinct_pillars=1, evidence_span_count=5),
        ]
        
        ranked1 = rank_candidates(candidates.copy())
        ranked2 = rank_candidates(candidates.copy())
        
        assert [c.name for c in ranked1] == [c.name for c in ranked2]


class TestCandidateGenerator:
    """Test candidate generation determinism."""

    def test_generate_candidate_all_edges(self):
        """Test baseline candidate generation."""
        edges = [
            {"edge_id": "e3", "from_node": "pillar:P001", "to_node": "pillar:P002"},
            {"edge_id": "e1", "from_node": "pillar:P002", "to_node": "pillar:P003"},
            {"edge_id": "e2", "from_node": "pillar:P003", "to_node": "pillar:P001"},
        ]
        
        cand = _generate_candidate_all_edges(edges, "test", max_edges=2)
        
        # Should be sorted by edge_id: e1, e2 (max 2)
        assert len(cand.used_edges) == 2
        assert cand.used_edges[0]["edge_id"] == "e1"
        assert cand.used_edges[1]["edge_id"] == "e2"

    def test_generate_candidate_by_diversity(self):
        """Test diversity-based candidate generation."""
        edges = [
            {"edge_id": "e1", "from_node": "pillar:P001", "to_node": "pillar:P002"},  # Cross-pillar
            {"edge_id": "e2", "from_node": "core_value:V001", "to_node": "sub_value:S001"},  # Value
            {"edge_id": "e3", "from_node": "pillar:P001", "to_node": "pillar:P001"},  # Same pillar
        ]
        
        cand = _generate_candidate_by_pillar_diversity(edges, "diverse", max_edges=2)
        
        # Cross-pillar should come first
        assert len(cand.used_edges) == 2
        assert cand.used_edges[0]["edge_id"] == "e1"

    def test_generate_candidate_by_relation_type(self):
        """Test relation-type based candidate generation."""
        edges = [
            {"edge_id": "e1", "relation_type": "ENABLES"},
            {"edge_id": "e2", "relation_type": "REINFORCES"},
            {"edge_id": "e3", "relation_type": "ENABLES"},
        ]
        
        cand = _generate_candidate_by_relation_type(edges, "ENABLES", "enables_cand")
        
        assert len(cand.used_edges) == 2
        assert all(e["relation_type"] == "ENABLES" for e in cand.used_edges)

    def test_generate_candidates_sync_empty(self):
        """Test sync generation with empty edges."""
        candidates = generate_candidates_sync([], "global_synthesis")
        assert candidates == []

    def test_generate_candidates_sync_deduplication(self):
        """Test that duplicate candidates are removed."""
        # Single edge will produce same candidate for multiple strategies
        edges = [
            {"edge_id": "e1", "from_node": "pillar:P001", "to_node": "pillar:P002", "relation_type": "ENABLES"},
        ]
        
        candidates = generate_candidates_sync(edges, "network_build", max_candidates=10)
        
        # Should be deduplicated
        edge_sets = [frozenset(e["edge_id"] for e in c.used_edges) for c in candidates]
        assert len(edge_sets) == len(set(edge_sets))

    def test_generate_candidates_sync_deterministic(self):
        """Test that sync generation is deterministic."""
        edges = [
            {"edge_id": f"e{i}", "from_node": f"pillar:P00{i%3+1}", 
             "to_node": f"pillar:P00{(i+1)%3+1}", "relation_type": "ENABLES"}
            for i in range(10)
        ]
        
        cands1 = generate_candidates_sync(edges, "global_synthesis")
        cands2 = generate_candidates_sync(edges, "global_synthesis")
        
        assert len(cands1) == len(cands2)
        for c1, c2 in zip(cands1, cands2):
            assert c1.name == c2.name
            assert len(c1.used_edges) == len(c2.used_edges)


class TestCriticLoop:
    """Test critic loop fixable-only triggers."""

    def test_fixable_failures_set(self):
        """Test fixable failures set is properly defined."""
        assert "MISSING_REQUIRED_ENTITY" in FIXABLE_FAILURES
        assert "EMPTY_REQUIRED_SECTION" in FIXABLE_FAILURES
        assert "MISSING_USED_EDGES" in FIXABLE_FAILURES
        assert "MISSING_BOUNDARY_SECTION" in FIXABLE_FAILURES

    def test_non_fixable_failures_set(self):
        """Test non-fixable failures set is properly defined."""
        assert "CITATION_VALIDITY_ERROR" in NON_FIXABLE_FAILURES
        assert "BINDING_PRUNE_FAILURE" in NON_FIXABLE_FAILURES
        assert "UNSUPPORTED_MUST_CITE" in NON_FIXABLE_FAILURES

    def test_should_trigger_critic_passing_contract(self):
        """Test no trigger when contract passes."""
        result = {"pass": True, "reasons": []}
        assert should_trigger_critic(result) is False

    def test_should_trigger_critic_fixable_only(self):
        """Test trigger when only fixable failures present."""
        result = {"pass": False, "reasons": ["MISSING_REQUIRED_ENTITY", "EMPTY_REQUIRED_SECTION"]}
        assert should_trigger_critic(result) is True

    def test_should_trigger_critic_non_fixable_present(self):
        """Test no trigger when non-fixable failure present."""
        result = {"pass": False, "reasons": ["MISSING_REQUIRED_ENTITY", "CITATION_VALIDITY_ERROR"]}
        assert should_trigger_critic(result) is False

    def test_should_trigger_critic_only_non_fixable(self):
        """Test no trigger when only non-fixable failures."""
        result = {"pass": False, "reasons": ["UNSUPPORTED_MUST_CITE"]}
        assert should_trigger_critic(result) is False

    def test_should_trigger_critic_empty_reasons(self):
        """Test no trigger when no reasons."""
        result = {"pass": False, "reasons": []}
        assert should_trigger_critic(result) is False

    def test_classify_contract_reasons_fixable(self):
        """Test classification of fixable reasons."""
        reasons = ["MISSING_REQUIRED_ENTITY: value:V001", "EMPTY_SECTION: حدود"]
        
        critique = _classify_contract_reasons(reasons)
        
        assert critique.needs_rewrite is True
        assert len(critique.fixable_issues) == 2
        assert len(critique.non_fixable_issues) == 0

    def test_classify_contract_reasons_mixed(self):
        """Test classification with mixed reasons."""
        reasons = ["MISSING_REQUIRED_ENTITY", "CITATION_VALIDITY_ERROR"]
        
        critique = _classify_contract_reasons(reasons)
        
        assert critique.needs_rewrite is False  # Non-fixable present
        assert len(critique.fixable_issues) == 1
        assert len(critique.non_fixable_issues) == 1

    def test_classify_contract_reasons_unknown_as_non_fixable(self):
        """Test that unknown reasons are treated as non-fixable."""
        reasons = ["UNKNOWN_ERROR_TYPE"]
        
        critique = _classify_contract_reasons(reasons)
        
        assert critique.needs_rewrite is False
        assert len(critique.non_fixable_issues) == 1


class TestCriticLoopIntegration:
    """Integration tests for critic loop behavior."""

    def test_critique_extracts_missing_entities(self):
        """Test that critique extracts missing entities from reasons."""
        reasons = ["MISSING_REQUIRED_ENTITY: pillar:P001"]
        
        critique = _classify_contract_reasons(reasons)
        
        assert len(critique.missing_entities) == 1
        assert "pillar:P001" in critique.missing_entities[0]

    def test_critique_extracts_missing_sections(self):
        """Test that critique extracts missing sections from reasons."""
        reasons = ["MISSING_BOUNDARY_SECTION", "EMPTY_SECTION: حدود الاستدلال"]
        
        critique = _classify_contract_reasons(reasons)
        
        assert len(critique.missing_sections) == 2

    def test_critic_not_invoked_for_citation_validity_errors(self):
        """GATE TEST: Critic loop must NOT run for citation validity errors."""
        # Citation validity errors must fail-closed immediately
        result = {"pass": False, "reasons": ["CITATION_VALIDITY_ERROR: invalid source"]}
        assert should_trigger_critic(result) is False
        
        result = {"pass": False, "reasons": ["INVALID_CITATION: CH_123"]}
        assert should_trigger_critic(result) is False

    def test_critic_not_invoked_for_binding_failures(self):
        """GATE TEST: Critic loop must NOT run for binding/prune failures."""
        # Binding failures must fail-closed immediately
        result = {"pass": False, "reasons": ["BINDING_PRUNE_FAILURE"]}
        assert should_trigger_critic(result) is False
        
        result = {"pass": False, "reasons": ["UNSUPPORTED_MUST_CITE: claim without evidence"]}
        assert should_trigger_critic(result) is False

    def test_critic_not_invoked_for_integrity_quarantine(self):
        """GATE TEST: Critic loop must NOT run for integrity/quarantine events."""
        # Hallucination/injection must fail-closed immediately
        result = {"pass": False, "reasons": ["HALLUCINATION_DETECTED"]}
        assert should_trigger_critic(result) is False
        
        result = {"pass": False, "reasons": ["INJECTION_DETECTED"]}
        assert should_trigger_critic(result) is False

    def test_critic_invoked_only_for_fixable_issues(self):
        """GATE TEST: Critic loop runs ONLY when ALL failures are fixable."""
        # All fixable - should trigger
        result = {"pass": False, "reasons": ["MISSING_REQUIRED_ENTITY", "EMPTY_REQUIRED_SECTION"]}
        assert should_trigger_critic(result) is True
        
        # Mixed fixable + non-fixable - should NOT trigger
        result = {"pass": False, "reasons": ["MISSING_REQUIRED_ENTITY", "CITATION_VALIDITY_ERROR"]}
        assert should_trigger_critic(result) is False
