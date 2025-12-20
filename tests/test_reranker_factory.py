from __future__ import annotations


def test_reranker_factory_disabled(monkeypatch) -> None:
    from apps.api.retrieve.reranker import NullReranker, create_reranker_from_env

    monkeypatch.delenv("RERANKER_ENABLED", raising=False)
    monkeypatch.delenv("RERANKER_MODEL_PATH", raising=False)
    rr = create_reranker_from_env()
    assert isinstance(rr, NullReranker)
    assert rr.is_enabled() is False

