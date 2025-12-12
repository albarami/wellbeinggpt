"""
End-to-end tests for the Ask API.

Tests the full pipeline including middleware and guardrails.
"""

import pytest
from unittest.mock import AsyncMock, patch

from apps.api.core.muhasibi_state_machine import (
    MuhasibiMiddleware,
    MuhasibiState,
    StateContext,
    create_middleware,
)
from apps.api.core.schemas import Confidence, Difficulty, EntityType


class TestMuhasibiMiddleware:
    """Tests for MuhasibiMiddleware."""

    @pytest.mark.asyncio
    async def test_process_returns_final_response(self):
        """Test that process returns a FinalResponse."""
        middleware = create_middleware()
        response = await middleware.process("ما هو الإيمان؟")

        assert response is not None
        assert hasattr(response, "listen_summary_ar")
        assert hasattr(response, "purpose")
        assert hasattr(response, "answer_ar")
        assert hasattr(response, "not_found")

    @pytest.mark.asyncio
    async def test_listen_state_normalizes_question(self):
        """Test that LISTEN state normalizes the question."""
        middleware = create_middleware()
        ctx = StateContext(question="مَا هُوَ الإِيمَانُ؟")

        await middleware._state_listen(ctx)

        assert ctx.normalized_question != ""
        assert ctx.listen_summary_ar != ""

    @pytest.mark.asyncio
    async def test_listen_state_extracts_keywords(self):
        """Test that LISTEN state extracts keywords."""
        middleware = create_middleware()
        ctx = StateContext(question="ما هو تعريف الإيمان في الإسلام؟")

        await middleware._state_listen(ctx)

        assert len(ctx.question_keywords) >= 1

    @pytest.mark.asyncio
    async def test_purpose_state_sets_constraints(self):
        """Test that PURPOSE state includes required constraints."""
        middleware = create_middleware()
        ctx = StateContext(question="سؤال")

        await middleware._state_purpose(ctx)

        assert ctx.purpose is not None
        assert "evidence_only" in ctx.purpose.constraints_ar
        assert "cite_every_claim" in ctx.purpose.constraints_ar
        assert "refuse_if_missing" in ctx.purpose.constraints_ar

    @pytest.mark.asyncio
    async def test_path_state_sets_plan(self):
        """Test that PATH state creates a plan."""
        middleware = create_middleware()
        ctx = StateContext(question="سؤال")

        await middleware._state_path(ctx)

        assert len(ctx.path_plan_ar) >= 1
        assert ctx.difficulty is not None

    @pytest.mark.asyncio
    async def test_account_state_detects_missing_evidence(self):
        """Test that ACCOUNT state detects missing evidence."""
        middleware = create_middleware()
        ctx = StateContext(question="سؤال")
        ctx.evidence_packets = []

        await middleware._state_account(ctx)

        assert ctx.not_found is True
        assert len(ctx.account_issues) >= 1

    @pytest.mark.asyncio
    async def test_interpret_state_refuses_when_no_evidence(self):
        """Test that INTERPRET state refuses when no evidence."""
        middleware = create_middleware()
        ctx = StateContext(question="سؤال")
        ctx.not_found = True

        await middleware._state_interpret(ctx)

        assert "لا يوجد" in ctx.answer_ar
        assert ctx.citations == []

    @pytest.mark.asyncio
    async def test_full_pipeline_no_evidence(self):
        """Test full pipeline with no evidence returns not_found."""
        middleware = create_middleware()
        response = await middleware.process("سؤال عشوائي غير موجود")

        assert response.not_found is True
        assert response.confidence == Confidence.LOW

    @pytest.mark.asyncio
    async def test_full_pipeline_with_entity_resolver(self):
        """Test pipeline with entity resolver."""
        from apps.api.retrieve.entity_resolver import EntityResolver

        resolver = EntityResolver()
        resolver.load_entities(
            pillars=[{"id": "P001", "name_ar": "الحياة الروحية"}],
            core_values=[{"id": "CV001", "name_ar": "الإيمان"}],
            sub_values=[],
        )

        middleware = create_middleware(entity_resolver=resolver)
        response = await middleware.process("ما هو الإيمان في الحياة الروحية؟")

        # Should detect entities
        assert "الإيمان" in response.listen_summary_ar or len(response.entities) > 0


class TestStateTransitions:
    """Tests for state machine transitions."""

    @pytest.mark.asyncio
    async def test_listen_to_purpose(self):
        """Test LISTEN transitions to PURPOSE."""
        middleware = create_middleware()
        ctx = StateContext(question="سؤال")

        next_state = await middleware._execute_state(MuhasibiState.LISTEN, ctx)

        assert next_state == MuhasibiState.PURPOSE

    @pytest.mark.asyncio
    async def test_purpose_to_path(self):
        """Test PURPOSE transitions to PATH."""
        middleware = create_middleware()
        ctx = StateContext(question="سؤال")

        next_state = await middleware._execute_state(MuhasibiState.PURPOSE, ctx)

        assert next_state == MuhasibiState.PATH

    @pytest.mark.asyncio
    async def test_account_to_interpret(self):
        """Test ACCOUNT transitions to INTERPRET."""
        middleware = create_middleware()
        ctx = StateContext(question="سؤال")
        ctx.evidence_packets = []

        next_state = await middleware._execute_state(MuhasibiState.ACCOUNT, ctx)

        assert next_state == MuhasibiState.INTERPRET


class TestEndToEndScenarios:
    """End-to-end scenario tests."""

    @pytest.mark.asyncio
    async def test_in_corpus_question_flow(self):
        """Test the flow for an in-corpus question."""
        # This would require mocked retriever with evidence
        middleware = create_middleware()

        # Simulate what would happen with evidence
        ctx = StateContext(question="ما هو الإيمان؟")
        ctx.evidence_packets = [
            {
                "chunk_id": "CH_000001",
                "entity_type": "core_value",
                "entity_id": "CV001",
                "chunk_type": "definition",
                "text_ar": "الإيمان هو التصديق بالقلب",
                "source_doc_id": "DOC_test",
                "source_anchor": "p1",
                "refs": [],
            }
        ]
        ctx.has_definition = True

        # Run account
        await middleware._state_account(ctx)
        assert ctx.citation_valid is True
        assert ctx.not_found is False

    @pytest.mark.asyncio
    async def test_out_of_corpus_question_flow(self):
        """Test the flow for an out-of-corpus question."""
        middleware = create_middleware()

        response = await middleware.process(
            "ما هي نظرية الكم في الفيزياء؟"  # Clearly out of scope
        )

        # Should return not_found
        assert response.not_found is True
        assert response.confidence == Confidence.LOW
        assert len(response.citations) == 0

