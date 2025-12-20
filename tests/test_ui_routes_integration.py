"""
UI routes integration tests.

Covers:
- POST /ask/ui
- GET /chunk/{chunk_id}
- GET /graph/expand
- GET /graph/path
- GET /graph/edge/{edge_id}/evidence
- POST /feedback
- GET /ask/runs/{request_id}
- GET /ask/runs/{request_id}/bundle

These tests require a live database (RUN_DB_TESTS=1) and a populated framework.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text


@pytest.fixture
async def client():
    from apps.api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_ask_ui_returns_required_fields(require_db, client):
    resp = await client.post(
        "/ask/ui",
        json={"question": "ما هي الركائز الخمس للحياة الطيبة؟", "language": "ar", "mode": "natural_chat", "engine": "muhasibi"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Required top-level fields
    for k in [
        "request_id",
        "latency_ms",
        "mode_used",
        "engine_used",
        "contract_outcome",
        "final",
        "citations_spans",
        "graph_trace",
        "muhasibi_trace",
    ]:
        assert k in data

    assert data["contract_outcome"] in {"PASS_FULL", "PASS_PARTIAL", "FAIL"}
    assert isinstance(data["citations_spans"], list)
    assert isinstance(data["graph_trace"].get("used_edges", []), list)
    assert isinstance(data["graph_trace"].get("argument_chains", []), list)
    assert isinstance(data["muhasibi_trace"], list)

    # Span resolution hard gate: offsets are either both ints or both null.
    for c in data["citations_spans"][:10]:
        ss = c.get("span_start")
        se = c.get("span_end")
        if ss is None or se is None:
            assert ss is None and se is None
            assert c.get("span_resolution_status") == "unresolved"
        else:
            assert isinstance(ss, int) and isinstance(se, int)
            assert 0 <= ss < se


@pytest.mark.asyncio
async def test_chunk_endpoint_expected_use(require_db, client):
    from apps.api.core.database import get_session

    async with get_session() as s:
        row = (await s.execute(text("SELECT chunk_id FROM chunk LIMIT 1"))).fetchone()
    assert row is not None

    resp = await client.get(f"/chunk/{row.chunk_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunk_id"] == row.chunk_id
    assert isinstance(data.get("text_ar", ""), str)


@pytest.mark.asyncio
async def test_chunk_endpoint_failure_case(require_db, client):
    resp = await client.get("/chunk/CH_DOES_NOT_EXIST")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_graph_expand_grounded_only_defaults(require_db, client):
    from apps.api.core.database import get_session

    async with get_session() as s:
        row = (await s.execute(text("SELECT id FROM pillar LIMIT 1"))).fetchone()
    assert row is not None

    resp = await client.get(
        "/graph/expand",
        params={"node_type": "pillar", "node_id": row.id, "depth": 2},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["grounded_only"] is True
    assert data["returned_nodes"] <= 2000
    assert data["returned_edges"] <= 5000

    # If relation_type is present, grounded_only implies has_evidence must be true.
    for e in data.get("edges", [])[:200]:
        if e.get("relation_type"):
            assert e.get("has_evidence") is True


@pytest.mark.asyncio
async def test_graph_edge_evidence_expected_use(require_db, client):
    from apps.api.core.database import get_session

    async with get_session() as s:
        row = (
            await s.execute(
                text(
                    """
                    SELECT s.edge_id::text AS edge_id
                    FROM edge_justification_span s
                    LIMIT 1
                    """
                )
            )
        ).fetchone()
    assert row is not None

    resp = await client.get(f"/graph/edge/{row.edge_id}/evidence")
    assert resp.status_code == 200
    data = resp.json()
    assert data["edge_id"] == row.edge_id
    assert isinstance(data["spans"], list)
    assert data["spans"], "Expected at least one span"


@pytest.mark.asyncio
async def test_graph_path_alias_expected_use(require_db, client):
    from apps.api.core.database import get_session

    async with get_session() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT p.id AS pillar_id, cv.id AS cv_id
                    FROM pillar p
                    JOIN core_value cv ON cv.pillar_id = p.id
                    LIMIT 1
                    """
                )
            )
        ).fetchone()

    if not row:
        pytest.skip("No pillar/core_value in DB")

    resp = await client.get(
        "/graph/path",
        params={
            "start_type": "pillar",
            "start_id": row.pillar_id,
            "target_type": "core_value",
            "target_id": row.cv_id,
            "max_depth": 4,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert isinstance(data["path"], list) and len(data["path"]) >= 2


@pytest.mark.asyncio
async def test_feedback_and_replay_bundle(require_db, client):
    # Run ask/ui first (creates ask_run best-effort)
    resp = await client.post(
        "/ask/ui",
        json={"question": "ما هي الحياة الروحية؟", "language": "ar", "mode": "answer", "engine": "muhasibi"},
    )
    assert resp.status_code == 200
    request_id = resp.json()["request_id"]
    assert isinstance(request_id, str) and request_id

    # Replay
    replay = await client.get(f"/ask/runs/{request_id}")
    assert replay.status_code == 200
    assert replay.json()["request_id"] == request_id

    # Feedback (edge case: allow missing rating)
    fb = await client.post("/feedback", json={"request_id": request_id, "rating": 1})
    assert fb.status_code == 200
    assert "stored" in fb.json()

    # Bundle
    bundle = await client.get(f"/ask/runs/{request_id}/bundle")
    assert bundle.status_code == 200
    b = bundle.json()
    assert "ask" in b and "debug_summary" in b
    assert b["ask"]["request_id"] == request_id


@pytest.mark.asyncio
async def test_feedback_failure_case_missing_request_id(require_db, client):
    resp = await client.post("/feedback", json={"rating": 1})
    assert resp.status_code == 400

