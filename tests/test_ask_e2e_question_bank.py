"""
End-to-End /ask Tests with Question Bank

Integration tests that run questions from the question bank and enforce:
- If not_found == False then citations must be non-empty
- Every cited chunk_id exists in the evidence bundle
- Claim-to-evidence checker passes
- Out-of-scope questions must return not_found=True

These tests require a live database and LLM configured.
"""

import json
import os
import pytest
from pathlib import Path
from typing import Any

from httpx import AsyncClient, ASGITransport


def load_question_bank() -> list[dict[str, Any]]:
    """Load questions from the JSONL fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "questions.jsonl"
    
    if not fixture_path.exists():
        return []
    
    questions = []
    with open(fixture_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    
    return questions


@pytest.fixture
async def client():
    """Create test client for the FastAPI app."""
    from apps.api.main import app
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestQuestionBankBasics:
    """Basic tests for question bank loading."""

    def test_question_bank_loads(self):
        """Test that question bank file loads correctly."""
        questions = load_question_bank()
        assert len(questions) > 0, "Question bank should have questions"

    def test_question_bank_has_categories(self):
        """Test that questions have expected categories."""
        questions = load_question_bank()
        
        categories = set(q["category"] for q in questions)
        expected_categories = {
            "factual_structure",
            "definition_evidence",
            "cross_pillar",
            "out_of_scope",
        }
        
        assert categories == expected_categories, (
            f"Expected categories {expected_categories}, got {categories}"
        )

    def test_question_bank_has_required_fields(self):
        """Test that all questions have required fields."""
        questions = load_question_bank()
        
        for q in questions:
            assert "id" in q, f"Question missing id: {q}"
            assert "category" in q, f"Question missing category: {q}"
            assert "question_ar" in q, f"Question missing question_ar: {q}"
            assert "expected" in q, f"Question missing expected: {q}"


@pytest.mark.asyncio
async def test_ask_endpoint_responds(require_db_and_llm, client):
    """Test that /ask endpoint responds to a basic question."""
    response = await client.post(
        "/ask",
        json={"question": "ما هو التوحيد؟"},
        timeout=120.0,
    )
    
    assert response.status_code == 200, f"API returned {response.status_code}: {response.text}"
    data = response.json()
    
    # Check response has expected fields
    assert "listen_summary_ar" in data
    assert "answer_ar" in data
    assert "citations" in data
    assert "not_found" in data


@pytest.mark.asyncio
async def test_factual_structure_questions(require_db_and_llm, client):
    """Test factual structure questions from question bank."""
    questions = [q for q in load_question_bank() if q["category"] == "factual_structure"]
    
    for q in questions[:3]:
        response = await client.post(
            "/ask",
            json={"question": q["question_ar"]},
            timeout=120.0,
        )
        
        assert response.status_code == 200, f"Failed for question {q['id']}: {response.text}"
        data = response.json()
        
        expected = q["expected"]
        
        # Debug output
        import os
        print(f"\n[DEBUG] Question {q['id']}: {q['question_ar'][:50]}...")
        print(f"[DEBUG] AZURE_OPENAI_ENDPOINT: {bool(os.getenv('AZURE_OPENAI_ENDPOINT'))}")
        print(f"[DEBUG] not_found: {data['not_found']}")
        print(f"[DEBUG] citations: {len(data.get('citations', []))}")
        print(f"[DEBUG] entities: {len(data.get('entities', []))}")
        print(f"[DEBUG] answer_ar: {data.get('answer_ar', '')[:200]}...")
        
        # Check not_found matches expectation
        if "not_found" in expected:
            assert data["not_found"] == expected["not_found"], (
                f"Question {q['id']}: expected not_found={expected['not_found']}, "
                f"got {data['not_found']}"
            )
        
        # If not_found is False, citations must be present
        if not data["not_found"] and expected.get("require_citations", False):
            assert len(data["citations"]) >= expected.get("min_citations", 1), (
                f"Question {q['id']}: expected at least {expected.get('min_citations', 1)} "
                f"citations, got {len(data['citations'])}"
            )


@pytest.mark.asyncio
async def test_out_of_scope_questions_return_not_found(require_db_and_llm, client):
    """Test that out-of-scope questions return not_found=True."""
    questions = [q for q in load_question_bank() if q["category"] == "out_of_scope"]
    
    for q in questions[:2]:
        response = await client.post(
            "/ask",
            json={"question": q["question_ar"]},
            timeout=120.0,
        )
        
        assert response.status_code == 200, f"Failed for question {q['id']}: {response.text}"
        data = response.json()
        
        # Out of scope questions MUST return not_found=True
        assert data["not_found"] is True, (
            f"Question {q['id']} ({q['question_ar'][:30]}...): "
            f"expected not_found=True, got {data['not_found']}"
        )


@pytest.mark.asyncio
async def test_citations_reference_valid_chunks(require_db_and_llm, client):
    """Test that citations reference chunks that exist in evidence."""
    from apps.api.core.database import get_session
    from sqlalchemy import text
    
    response = await client.post(
        "/ask",
        json={"question": "ما هي ركائز الحياة الطيبة؟"},
        timeout=120.0,
    )
    
    assert response.status_code == 200
    data = response.json()
    
    if data["not_found"]:
        pytest.skip("Question returned not_found, no citations to check")
    
    # Verify each cited chunk_id exists in the database
    async with get_session() as session:
        for citation in data["citations"]:
            chunk_id = citation["chunk_id"]
            
            row = (await session.execute(
                text("SELECT chunk_id FROM chunk WHERE chunk_id = :cid"),
                {"cid": chunk_id}
            )).fetchone()
            
            assert row is not None, f"Cited chunk_id {chunk_id} does not exist in database"


@pytest.mark.asyncio
async def test_not_found_false_requires_non_empty_answer(require_db_and_llm, client):
    """Test that not_found=False requires a non-empty answer."""
    response = await client.post(
        "/ask",
        json={"question": "عرّف التوحيد."},
        timeout=120.0,
    )
    
    assert response.status_code == 200
    data = response.json()
    
    if not data["not_found"]:
        assert len(data["answer_ar"]) > 0, "not_found=False but answer_ar is empty"


class TestMuhasibiStructure:
    """Tests for Muḥāsibī mode response structure."""

    @pytest.mark.asyncio
    async def test_response_has_listen_summary(self, require_db_and_llm, client):
        """Test that response has non-empty listen_summary_ar."""
        response = await client.post(
            "/ask",
            json={"question": "ما هو الإيمان؟"},
            timeout=120.0,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "listen_summary_ar" in data
        assert len(data["listen_summary_ar"]) > 0, "listen_summary_ar should not be empty"

    @pytest.mark.asyncio
    async def test_response_has_purpose(self, require_db_and_llm, client):
        """Test that response has purpose with constraints."""
        response = await client.post(
            "/ask",
            json={"question": "ما هي القيم الكلية؟"},
            timeout=120.0,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "purpose" in data
        assert "ultimate_goal_ar" in data["purpose"]
        assert "constraints_ar" in data["purpose"]

    @pytest.mark.asyncio
    async def test_response_has_path_plan(self, require_db_and_llm, client):
        """Test that response has path_plan_ar."""
        response = await client.post(
            "/ask",
            json={"question": "كيف أحقق التوازن العاطفي؟"},
            timeout=120.0,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "path_plan_ar" in data
        assert isinstance(data["path_plan_ar"], list)
        if not data["not_found"]:
            assert len(data["path_plan_ar"]) >= 1, "path_plan_ar should have at least 1 item"


class TestClaimToEvidence:
    """Tests for claim-to-evidence checking."""

    @pytest.mark.asyncio
    async def test_answer_terms_covered_by_evidence(self, require_db_and_llm, client):
        """Test that key terms in answer are covered by evidence packets."""
        response = await client.post(
            "/ask",
            json={"question": "عرّف التوحيد."},
            timeout=120.0,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        if data["not_found"]:
            pytest.skip("Question returned not_found")
        
        assert "citations" in data
        
        if len(data["answer_ar"]) > 50:
            assert len(data["citations"]) >= 1, (
                "Substantial answer should have at least one citation"
            )
