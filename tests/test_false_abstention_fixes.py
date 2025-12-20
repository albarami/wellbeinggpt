"""
Tests for false abstention fixes.

These tests verify that the ACCOUNT gate correctly bypasses relevance checks
for system policy questions and broad guidance questions.

Root cause: apply_question_evidence_relevance_gate() was setting not_found=True
when matched==0, even when RETRIEVE found strong evidence packets.

Fixes:
1. SYSTEM_LIMITS_POLICY intent bypasses relevance gate
2. GUIDANCE_FRAMEWORK_CHAT intent bypasses relevance gate  
3. Seed retrieval floor ensures broad intents have pillar definitions
"""

import pytest
from unittest.mock import MagicMock, patch

from apps.api.core.muhasibi_listen import (
    _is_system_limits_policy_intent,
    _is_guidance_framework_chat_intent,
)
from apps.api.core.muhasibi_account import apply_question_evidence_relevance_gate
from apps.api.retrieve.normalize_ar import normalize_for_matching


class TestSystemLimitsPolicyIntentDetection:
    """Test detection of system policy questions."""

    def test_detects_bounds_question(self):
        """bound-009: ما حدود الربط بين الركائز غير المنصوص عليها؟"""
        q = normalize_for_matching("ما حدود الربط بين الركائز غير المنصوص عليها؟")
        assert _is_system_limits_policy_intent(q) is True

    def test_detects_methodology_question(self):
        """Questions about system methodology."""
        q = normalize_for_matching("كيف يربط النظام بين الركائز؟")
        assert _is_system_limits_policy_intent(q) is True

    def test_detects_not_covered_question(self):
        """Questions about what framework doesn't cover."""
        q = normalize_for_matching("ما لا ينص عليه الإطار؟")
        assert _is_system_limits_policy_intent(q) is True

    def test_rejects_normal_question(self):
        """Normal framework questions should not match."""
        q = normalize_for_matching("ما هي الركيزة الروحية؟")
        assert _is_system_limits_policy_intent(q) is False

    def test_rejects_fiqh_question(self):
        """Fiqh questions should not match."""
        q = normalize_for_matching("ما حكم صلاة الجمعة؟")
        assert _is_system_limits_policy_intent(q) is False


class TestGuidanceFrameworkChatIntentDetection:
    """Test detection of broad guidance questions."""

    def test_detects_meaning_loss_question(self):
        """chat-011: أشعر بفقدان المعنى في حياتي، ماذا أفعل؟"""
        q = normalize_for_matching("أشعر بفقدان المعنى في حياتي، ماذا أفعل؟")
        assert _is_guidance_framework_chat_intent(q, "natural_chat") is True

    def test_detects_self_improvement_question(self):
        """chat-019: أريد أن أكون شخصًا أفضل، من أين أبدأ؟"""
        q = normalize_for_matching("أريد أن أكون شخصًا أفضل، من أين أبدأ؟")
        assert _is_guidance_framework_chat_intent(q, "natural_chat") is True

    def test_detects_where_to_start(self):
        """Generic 'where to start' questions."""
        q = normalize_for_matching("كيف أبدأ في تحسين حياتي؟")
        assert _is_guidance_framework_chat_intent(q, "natural_chat") is True

    def test_requires_natural_chat_mode(self):
        """Must be in natural_chat mode."""
        q = normalize_for_matching("أشعر بفقدان المعنى في حياتي، ماذا أفعل؟")
        assert _is_guidance_framework_chat_intent(q, "answer") is False
        assert _is_guidance_framework_chat_intent(q, "debate") is False

    def test_rejects_specific_entity_question(self):
        """Questions with specific entities should not match."""
        q = normalize_for_matching("ما هو الإيمان وكيف يرتبط بالتوازن؟")
        assert _is_guidance_framework_chat_intent(q, "natural_chat") is False


class TestAccountGateBypass:
    """Test that ACCOUNT gate bypasses relevance check for new intents."""

    def _make_ctx(self, intent_type: str = None, mode: str = "answer"):
        """Create a mock context."""
        ctx = MagicMock()
        ctx.question = "test question"
        ctx.not_found = False
        ctx.citation_valid = True
        ctx.account_issues = []
        ctx.evidence_packets = [
            {"chunk_id": "1", "text_ar": "evidence text", "chunk_type": "definition"}
            for _ in range(5)
        ]
        ctx.detected_entities = []
        ctx.mode = mode
        
        if intent_type:
            ctx.intent = {
                "intent_type": intent_type,
                "is_in_scope": True,
                "bypass_relevance_gate": intent_type == "system_limits_policy",
                "requires_seed_retrieval": intent_type == "guidance_framework_chat",
            }
        else:
            ctx.intent = None
            
        return ctx

    def test_bypasses_for_system_limits_policy(self):
        """SYSTEM_LIMITS_POLICY should bypass relevance gate."""
        ctx = self._make_ctx(intent_type="system_limits_policy")
        apply_question_evidence_relevance_gate(ctx)
        
        # Should not set not_found
        assert ctx.not_found is False

    def test_bypasses_for_guidance_framework_chat(self):
        """GUIDANCE_FRAMEWORK_CHAT should bypass relevance gate."""
        ctx = self._make_ctx(intent_type="guidance_framework_chat", mode="natural_chat")
        apply_question_evidence_relevance_gate(ctx)
        
        # Should not set not_found
        assert ctx.not_found is False

    def test_bypasses_for_global_synthesis(self):
        """global_synthesis should bypass relevance gate."""
        ctx = self._make_ctx(intent_type="global_synthesis")
        apply_question_evidence_relevance_gate(ctx)
        
        # Should not set not_found
        assert ctx.not_found is False

    def test_bypasses_for_cross_pillar(self):
        """cross_pillar should bypass relevance gate."""
        ctx = self._make_ctx(intent_type="cross_pillar")
        apply_question_evidence_relevance_gate(ctx)
        
        # Should not set not_found
        assert ctx.not_found is False

    def test_bypasses_for_natural_chat_with_seeds(self):
        """natural_chat with seed packets should bypass relevance gate."""
        ctx = self._make_ctx(mode="natural_chat")
        # Add pillar seed packets
        ctx.evidence_packets = [
            {"chunk_id": str(i), "text_ar": f"ركيزة {i}", "entity_type": "pillar"}
            for i in range(5)
        ]
        apply_question_evidence_relevance_gate(ctx)
        
        # Should not set not_found
        assert ctx.not_found is False

    def test_does_not_bypass_for_normal_question_without_matches(self):
        """Normal questions without entity matches should still be gated."""
        ctx = self._make_ctx(mode="answer")
        ctx.question = "سؤال لا علاقة له بالأدلة"
        ctx.evidence_packets = [
            {"chunk_id": "1", "text_ar": "نص مختلف تماما", "chunk_type": "evidence"}
        ]
        ctx.intent = None
        ctx.detected_entities = []
        
        apply_question_evidence_relevance_gate(ctx)
        
        # Should set not_found because no terms match
        assert ctx.not_found is True


class TestRegressionQuestions:
    """
    Integration tests for the three regression questions.
    
    These tests verify the full intent detection + gate bypass flow.
    """

    def test_bound_009_intent_detection(self):
        """bound-009 should be detected as system_limits_policy."""
        q = normalize_for_matching("ما حدود الربط بين الركائز غير المنصوص عليها؟")
        assert _is_system_limits_policy_intent(q) is True

    def test_chat_011_intent_detection(self):
        """chat-011 should be detected as guidance_framework_chat in natural_chat mode."""
        q = normalize_for_matching("أشعر بفقدان المعنى في حياتي، ماذا أفعل؟")
        assert _is_guidance_framework_chat_intent(q, "natural_chat") is True

    def test_chat_019_intent_detection(self):
        """chat-019 should be detected as guidance_framework_chat in natural_chat mode."""
        q = normalize_for_matching("أريد أن أكون شخصًا أفضل، من أين أبدأ؟")
        assert _is_guidance_framework_chat_intent(q, "natural_chat") is True

