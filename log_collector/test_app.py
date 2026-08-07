from __future__ import annotations

import asyncio
import json

import httpx

from log_collector.app import create_app


VALID_RECORD = {
    "schema_version": "1.0",
    "timestamp": "2026-01-01T00:00:00Z",
    "level": "ERROR",
    "service": "payment-service",
    "host": "payment-service-1",
    "message": "Payment authorization timed out",
    "trace_id": "trace-test-001",
    "metadata": {"order_id": "order-1"},
}


class FakePipeline:
    def __init__(self) -> None:
        self.entries: list[tuple[str, dict[str, str]]] = []

    def xadd(self, stream: str, fields: dict[str, str]) -> None:
        self.entries.append((stream, fields))

    async def execute(self) -> list[str]:
        return [f"{index}-0" for index, _ in enumerate(self.entries, start=1)]


class FakeRedis:
    def __init__(self) -> None:
        self.pipelines: list[FakePipeline] = []

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        assert transaction is True
        pipeline = FakePipeline()
        self.pipelines.append(pipeline)
        return pipeline


async def _post(redis: FakeRedis, payload: object) -> httpx.Response:
    app = create_app(redis)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            return await client.post("/ingest", json=payload)


def test_ingest_accepts_one_record_and_writes_one_stream_entry() -> None:
    redis = FakeRedis()
    response = asyncio.run(_post(redis, VALID_RECORD))

    assert response.status_code == 202
    assert response.json()["accepted"] == 1
    entry = redis.pipelines[0].entries[0]
    assert entry[0] == "sift-logs"
    assert json.loads(entry[1]["record"]) == VALID_RECORD


def test_ingest_accepts_a_batch() -> None:
    redis = FakeRedis()
    second_record = {**VALID_RECORD, "trace_id": "trace-test-002"}

    response = asyncio.run(_post(redis, [VALID_RECORD, second_record]))

    assert response.status_code == 202
    assert response.json()["accepted"] == 2
    assert len(redis.pipelines[0].entries) == 2


def test_ingest_rejects_malformed_batch_without_writing() -> None:
    redis = FakeRedis()
    malformed = {**VALID_RECORD, "level": "NOT_A_LEVEL"}

    response = asyncio.run(_post(redis, [VALID_RECORD, malformed]))

    assert response.status_code == 422
    assert redis.pipelines == []
