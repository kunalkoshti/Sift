import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest

from log_api.app import ApiConfig, create_app
from log_api.retriever import RetrievedChunk


def _set_minimum_api_config(monkeypatch):
    values = {
        "POSTGRES_DSN": "postgresql://example/sift",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "RETRIEVAL_TOP_K": "5",
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": "test-model",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


class FakeRAGService:
    async def log_level_stats(self):
        return {
            "status": "ready",
            "last_indexed": datetime(2026, 10, 23, 2, 30, tzinfo=timezone.utc),
            "raw_logs": 1117,
            "embedded_chunks": 60,
            "intervals": [
                {
                    "start": datetime(2025, 12, 31, 23, 58, tzinfo=timezone.utc),
                    "end": datetime(2025, 12, 31, 23, 59, tzinfo=timezone.utc),
                    "counts": {"INFO": 4, "WARN": 2, "ERROR": 1, "CRITICAL": 0},
                    "total": 7,
                }
            ],
        }

    async def ask(self, question: str):
        return {
            "answer": f"Answer based on logs for: {question}",
            "retrieved_chunks": [
                RetrievedChunk(
                    id=uuid4(),
                    window_start=datetime(2025, 12, 31, 23, 59, tzinfo=timezone.utc),
                    window_end=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
                    sub_index=0,
                    services=["payment-service"],
                    trace_ids=["trace-payment-timeout-001"],
                    content="Payment authorization timed out",
                    cosine_distance=0.2,
                    cosine_similarity=0.8,
                )
            ],
        }


def test_health_and_ask_endpoint():
    async def exercise() -> httpx.Response:
        app = create_app(FakeRAGService())
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                health = await client.get("/health")
                assert health.json() == {"status": "ok"}
                stats = await client.get("/stats/log-levels")
                assert stats.status_code == 200
                assert stats.json()["raw_logs"] == 1117
                assert stats.json()["intervals"][0]["counts"]["WARN"] == 2
                return await client.post(
                    "/ask",
                    json={"question": "why did payments fail"},
                )

    response = asyncio.run(exercise())

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Answer based on logs")
    assert body["response_type"] == "supported"
    assert len(body["retrieved_chunks"]) == 1
    assert body["retrieved_chunks"][0]["trace_ids"] == ["trace-payment-timeout-001"]


def test_api_config_reads_database_pool_sizes(monkeypatch):
    _set_minimum_api_config(monkeypatch)
    monkeypatch.setenv("DB_POOL_MIN_SIZE", "1")
    monkeypatch.setenv("DB_POOL_MAX_SIZE", "3")

    config = ApiConfig.from_env()

    assert config.db_pool_min_size == 1
    assert config.db_pool_max_size == 3


def test_api_config_rejects_invalid_database_pool_sizes(monkeypatch):
    _set_minimum_api_config(monkeypatch)
    monkeypatch.setenv("DB_POOL_MIN_SIZE", "4")
    monkeypatch.setenv("DB_POOL_MAX_SIZE", "3")

    with pytest.raises(RuntimeError, match="DB_POOL_MAX_SIZE"):
        ApiConfig.from_env()


def test_blank_question_is_rejected():
    async def exercise() -> httpx.Response:
        app = create_app(FakeRAGService())
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.post("/ask", json={"question": "   "})

    response = asyncio.run(exercise())

    assert response.status_code == 422


def test_cors_allows_configured_frontend_origin(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://frontend.example")

    async def exercise() -> httpx.Response:
        app = create_app(FakeRAGService())
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.options(
                    "/ask",
                    headers={
                        "Origin": "https://frontend.example",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    },
                )

    response = asyncio.run(exercise())

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://frontend.example"


def test_cors_does_not_allow_unconfigured_origin(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://frontend.example")

    async def exercise() -> httpx.Response:
        app = create_app(FakeRAGService())
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.options(
                    "/ask",
                    headers={
                        "Origin": "https://other.example",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    },
                )

    response = asyncio.run(exercise())

    assert "access-control-allow-origin" not in response.headers
