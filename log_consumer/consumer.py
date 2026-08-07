"""Drain canonical log records from Redis into Postgres raw_logs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any

import asyncpg
from dotenv import load_dotenv
from pydantic import ValidationError
from redis.asyncio import Redis, from_url
from redis.exceptions import ResponseError

from sift_common.schema import LogRecord


ROOT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
SCHEMA_FILE = Path(__file__).resolve().parents[1] / "db" / "schema.sql"
load_dotenv(ROOT_ENV_FILE)

logger = logging.getLogger(__name__)

INSERT_RAW_LOG_SQL = """
INSERT INTO raw_logs (
  schema_version,
  timestamp,
  level,
  service,
  host,
  message,
  trace_id,
  metadata,
  redis_entry_id
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
ON CONFLICT (redis_entry_id) DO NOTHING
"""


@dataclass(frozen=True)
class ConsumerConfig:
    redis_url: str
    postgres_dsn: str
    stream: str
    group: str
    consumer: str
    count: int
    block_ms: int
    group_start_id: str

    @classmethod
    def from_env(cls) -> ConsumerConfig:
        def required(name: str) -> str:
            value = os.getenv(name)
            if not value:
                raise RuntimeError(f"{name} must be configured in .env or the environment")
            return value

        return cls(
            redis_url=required("REDIS_URL"),
            postgres_dsn=required("POSTGRES_DSN"),
            stream=required("REDIS_STREAM"),
            group=os.getenv("REDIS_GROUP", "sift-log-consumers"),
            consumer=os.getenv("REDIS_CONSUMER", "consumer-1"),
            count=int(os.getenv("REDIS_READ_COUNT", "100")),
            block_ms=int(os.getenv("REDIS_BLOCK_MS", "5000")),
            group_start_id=os.getenv("REDIS_GROUP_START_ID", "0"),
        )


def load_table_schema(path: Path = SCHEMA_FILE) -> str:
    return path.read_text(encoding="utf-8")


async def ensure_database_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection:
        await connection.execute(load_table_schema())


async def ensure_consumer_group(redis: Redis, config: ConsumerConfig) -> None:
    try:
        await redis.xgroup_create(
            name=config.stream,
            groupname=config.group,
            id=config.group_start_id,
            mkstream=True,
        )
        logger.info("created Redis consumer group %s on %s", config.group, config.stream)
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise
        logger.info("Redis consumer group %s already exists", config.group)


def parse_record(fields: dict[str, Any]) -> LogRecord:
    record_json = fields.get("record")
    if record_json is None:
        raise ValueError("Redis entry is missing the record field")
    return LogRecord.model_validate_json(record_json)


async def insert_record(
    pool: asyncpg.Pool,
    redis_entry_id: str,
    record: LogRecord,
) -> None:
    async with pool.acquire() as connection:
        await connection.execute(
            INSERT_RAW_LOG_SQL,
            record.schema_version,
            record.timestamp,
            record.level,
            record.service,
            record.host,
            record.message,
            record.trace_id,
            json.dumps(record.metadata),
            redis_entry_id,
        )


async def process_entry(
    redis: Redis,
    pool: asyncpg.Pool,
    config: ConsumerConfig,
    redis_entry_id: str,
    fields: dict[str, Any],
) -> bool:
    """Persist and acknowledge one entry; leave it pending on any failure."""

    try:
        record = parse_record(fields)
        await insert_record(pool, redis_entry_id, record)
        await redis.xack(config.stream, config.group, redis_entry_id)
        return True
    except (ValidationError, ValueError) as exc:
        logger.exception("invalid Redis entry %s; leaving it pending: %s", redis_entry_id, exc)
    except Exception:
        logger.exception("failed to persist Redis entry %s; leaving it pending", redis_entry_id)
    return False


async def read_and_process(
    redis: Redis,
    pool: asyncpg.Pool,
    config: ConsumerConfig,
    start_id: str,
    *,
    block_ms: int | None,
) -> int:
    messages = await redis.xreadgroup(
        groupname=config.group,
        consumername=config.consumer,
        streams={config.stream: start_id},
        count=config.count,
        block=block_ms,
    )
    processed = 0
    for _stream_name, entries in messages:
        for entry_id, fields in entries:
            if await process_entry(redis, pool, config, entry_id, fields):
                processed += 1
    return processed


async def run(config: ConsumerConfig | None = None) -> None:
    config = config or ConsumerConfig.from_env()
    redis = from_url(config.redis_url, decode_responses=True)
    pool = await asyncpg.create_pool(config.postgres_dsn)
    try:
        await ensure_database_schema(pool)
        await ensure_consumer_group(redis, config)

        # Reclaim entries pending for this stable consumer name after a restart.
        # A missing block value makes this a non-blocking drain. Redis treats
        # BLOCK 0 as "wait forever", which would prevent normal consumption
        # when there are no pending entries to recover.
        await read_and_process(redis, pool, config, "0", block_ms=None)
        logger.info(
            "consuming stream=%s group=%s consumer=%s",
            config.stream,
            config.group,
            config.consumer,
        )
        while True:
            await read_and_process(redis, pool, config, ">", block_ms=config.block_ms)
    finally:
        await pool.close()
        await redis.aclose()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("consumer stopped")


if __name__ == "__main__":
    main()
