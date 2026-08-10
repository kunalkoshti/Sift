from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

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
    with TestClient(create_app(FakeRAGService())) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/ask", json={"question": "why did payments fail"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Answer based on logs")
    assert len(body["retrieved_chunks"]) == 1
    assert body["retrieved_chunks"][0]["trace_ids"] == ["trace-payment-timeout-001"]


def test_blank_question_is_rejected():
    with TestClient(create_app(FakeRAGService())) as client:
        response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422
