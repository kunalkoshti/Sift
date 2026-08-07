"""Validate log records and enqueue them onto a Redis stream."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from redis.asyncio import Redis, from_url
from redis.exceptions import RedisError

from sift_common.schema import LogRecord


ROOT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ROOT_ENV_FILE)


class IngestResponse(BaseModel):
    accepted: int = Field(ge=1)
    stream: str
    entry_ids: list[str]


def create_app(redis_client: Redis | Any | None = None) -> FastAPI:
    """Create the collector application with an optional injectable Redis client."""

    redis_url = os.getenv("REDIS_URL")
    stream_name = os.getenv("REDIS_STREAM")
    if not redis_url or not stream_name:
        raise RuntimeError("REDIS_URL and REDIS_STREAM must be configured in .env or the environment")
    owns_redis_client = redis_client is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.redis = (
            redis_client
            if redis_client is not None
            else from_url(redis_url, decode_responses=True)
        )
        try:
            yield
        finally:
            if owns_redis_client:
                await app.state.redis.aclose()

    app = FastAPI(
        title="Sift Log Collector",
        description="Validates canonical log records and queues them on Redis.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest", response_model=IngestResponse, status_code=202)
    async def ingest(payload: LogRecord | list[LogRecord]) -> IngestResponse:
        """Validate one or more records and enqueue them atomically."""

        records = payload if isinstance(payload, list) else [payload]
        if not records:
            raise HTTPException(status_code=422, detail="ingest payload must not be an empty list")

        redis_client_for_request: Redis = app.state.redis
        pipeline = redis_client_for_request.pipeline(transaction=True)
        for record in records:
            pipeline.xadd(
                stream_name,
                {"record": record.model_dump_json()},
            )

        try:
            entry_ids = await pipeline.execute()
        except RedisError as exc:
            raise HTTPException(status_code=503, detail="Redis is unavailable") from exc

        return IngestResponse(
            accepted=len(records),
            stream=stream_name,
            entry_ids=[str(entry_id) for entry_id in entry_ids],
        )

    return app


app = create_app()
