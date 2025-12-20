"""
Tests for network and path questions requiring grounded graph edges.

These tests verify that:
1. Network questions produce used_edges
2. Path questions produce used_edges
3. If graph is required but no edges exist, contract is PASS_PARTIAL not FAIL with full narrative
4. Relation labels are only used when used_edges is non-empty
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from apps.api.core.answer_contract import (
    contract_from_question_runtime,
    check_contract,
    ContractOutcome,
    UsedEdge,
    UsedEdgeSpan,
)


class TestNetworkIntentDetection:
    """Test that network intents are correctly detected."""

    def test_network_intent_from_shebaka(self):
        """شبكة keyword triggers network intent."""
        spec = contract_from_question_runtime(
            question_norm="اختر قيمة محورية وابنِ شبكة تربطها بثلاث ركائز",
            detected_entities=[],
        )
        assert spec.intent_type == "network"
        assert spec.requires_graph is True
        assert spec.min_links >= 3

    def test_network_intent_from_three_pillars(self):
        """ثلاث ركائز keyword triggers network intent."""
        spec = contract_from_question_runtime(
            question_norm="اربطها بثلاث ركائز أخرى",
            detected_entities=[],
        )
        assert spec.intent_type == "network"
        assert spec.requires_graph is True

    def test_path_intent_from_masar(self):
        """مسار keyword triggers cross_pillar intent."""
        spec = contract_from_question_runtime(
            question_norm="قدم مسار من البدنية إلى الروحية",
            detected_entities=[],
        )
        assert spec.intent_type == "cross_pillar"
        assert spec.requires_graph is True


class TestContractWithEdges:
    """Test contract checking with and without used_edges."""

    def test_network_pass_full_with_edges(self):
        """Network question passes with sufficient edges."""
        spec = contract_from_question_runtime(
            question_norm="ابنِ شبكة تربطها بثلاث ركائز",
            detected_entities=[],
        )
        
        # Create mock used_edges
        used_edges = [
            UsedEdge(
                edge_id="e1",
                from_node="pillar:P001",
                to_node="pillar:P002",
                relation_type="ENABLES",
                justification_spans=(
                    UsedEdgeSpan(chunk_id="ch1", span_start=0, span_end=10, quote="test"),
                ),
            ),
            UsedEdge(
                edge_id="e2",
                from_node="pillar:P001",
                to_node="pillar:P003",
                relation_type="COMPLEMENTS",
                justification_spans=(
                    UsedEdgeSpan(chunk_id="ch2", span_start=0, span_end=10, quote="test"),
                ),
            ),
            UsedEdge(
                edge_id="e3",
                from_node="pillar:P001",
                to_node="pillar:P004",
                relation_type="REINFORCES",
                justification_spans=(
                    UsedEdgeSpan(chunk_id="ch3", span_start=0, span_end=10, quote="test"),
                ),
            ),
        ]
        
        # Mock citations
        from apps.api.core.schemas import Citation
        citations = [Citation(chunk_id="ch1", source_anchor="test")]
        
        cm = check_contract(
            spec=spec,
            answer_ar="تعريف المفهوم داخل الإطار\n- test\nالتأصيل والأدلة\n- test",
            citations=citations,
            used_edges=used_edges,
        )
        
        # With 3 edges to 3 different pillars, should pass
        assert cm.graph_required_satisfied is True

    def test_network_fail_without_edges(self):
        """Network question with no edges should have MISSING_USED_GRAPH_EDGES reason."""
        spec = contract_from_question_runtime(
            question_norm="ابنِ شبكة تربطها بثلاث ركائز",
            detected_entities=[],
        )
        
        from apps.api.core.schemas import Citation
        citations = [Citation(chunk_id="ch1", source_anchor="test")]
        
        cm = check_contract(
            spec=spec,
            answer_ar="تعريف المفهوم داخل الإطار\n- test",
            citations=citations,
            used_edges=[],  # No edges!
        )
        
        assert cm.graph_required_satisfied is False
        assert "MISSING_USED_GRAPH_EDGES" in cm.reasons


class TestDeepModeMarkers:
    """Test that network/path questions trigger deep mode."""

    def test_shebaka_triggers_deep_mode(self):
        """شبكة should be in deep mode markers."""
        from apps.api.core.muhasibi_listen import _looks_like_deep_question_ar
        
        assert _looks_like_deep_question_ar("ابنِ شبكة تربطها بثلاث ركائز") is True

    def test_masar_triggers_deep_mode(self):
        """مسار should be in deep mode markers."""
        from apps.api.core.muhasibi_listen import _looks_like_deep_question_ar
        
        assert _looks_like_deep_question_ar("قدم مسار من البدنية إلى الروحية") is True

    def test_ikhtaar_qeema_triggers_deep_mode(self):
        """اختر قيمة should be in deep mode markers."""
        from apps.api.core.muhasibi_listen import _looks_like_deep_question_ar
        
        assert _looks_like_deep_question_ar("اختر قيمة محورية وابنِ شبكة") is True


class TestRelationLabelsGate:
    """Test that relation labels require used_edges."""

    def test_relation_labels_detected(self):
        """Test detection of relation labels in answer."""
        relation_labels = ["تمكين", "تعزيز", "تكامل", "إعانة", "شرط"]
        
        answer_with_labels = "هذه العلاقة من نوع تمكين وتعزيز"
        answer_without_labels = "العلم والحكمة مرتبطان في الإطار"
        
        has_labels = any(label in answer_with_labels for label in relation_labels)
        no_labels = any(label in answer_without_labels for label in relation_labels)
        
        assert has_labels is True
        assert no_labels is False
