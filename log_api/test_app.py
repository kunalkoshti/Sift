import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import httpx

from log_api.app import create_app
from log_api.retriever import RetrievedChunk


class FakeRAGService:
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
                return await client.post(
                    "/ask",
                    json={"question": "why did payments fail"},
                )

    response = asyncio.run(exercise())

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Answer based on logs")
    assert len(body["retrieved_chunks"]) == 1
    assert body["retrieved_chunks"][0]["trace_ids"] == ["trace-payment-timeout-001"]


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
