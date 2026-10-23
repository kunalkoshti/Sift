"""Build deterministic time-windowed chunks and embed them into Postgres."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import asyncpg
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_FILE = ROOT_DIR / "db" / "schema.sql"
EMBEDDING_DIMENSION = 384

load_dotenv(ROOT_DIR / ".env")
logger = logging.getLogger(__name__)

RAW_LOGS_SQL = """
SELECT id, timestamp, level, service, message, trace_id
FROM raw_logs
ORDER BY timestamp, id
"""

RAW_LOGS_AFTER_SQL = """
SELECT id, timestamp, level, service, message, trace_id
FROM raw_logs
WHERE id > $1
ORDER BY id
"""

RAW_LOGS_WINDOW_SQL = """
SELECT id, timestamp, level, service, message, trace_id
FROM raw_logs
WHERE trace_id IS NOT DISTINCT FROM $1::text
  AND timestamp >= $2
  AND timestamp < $3
ORDER BY timestamp, id
"""

CHECKPOINT_SQL = """
SELECT last_raw_log_id
FROM embedder_state
WHERE worker_name = $1
"""

UPSERT_CHECKPOINT_SQL = """
INSERT INTO embedder_state (worker_name, last_raw_log_id, updated_at)
VALUES ($1, $2, now())
ON CONFLICT (worker_name) DO UPDATE SET
  last_raw_log_id = EXCLUDED.last_raw_log_id,
  updated_at = now()
"""

INSERT_CHUNK_SQL = """
INSERT INTO chunks (
  chunk_type,
  window_start,
  window_end,
  sub_index,
  services,
  trace_ids,
  max_level,
  content,
  embedding,
  raw_log_ids
) VALUES (
  $1,
  $2,
  $3,
  $4,
  $5::text[],
  $6::text[],
  $7,
  $8,
  $9::vector,
  $10::bigint[]
)
"""


@dataclass(frozen=True)
class EmbedderConfig:
    postgres_dsn: str
    model_name: str
    window_seconds: int
    max_records_per_chunk: int
    embed_batch_size: int
    mode: str
    poll_seconds: float
    worker_name: str

    @classmethod
    def from_env(cls) -> EmbedderConfig:
        def required(name: str) -> str:
            value = os.getenv(name)
            if not value:
                raise RuntimeError(f"{name} must be configured in .env or the environment")
            return value

        mode = os.getenv("EMBEDDER_MODE", "batch").strip().lower()
        if mode not in {"batch", "live"}:
            raise RuntimeError("EMBEDDER_MODE must be 'batch' or 'live'")

        poll_seconds = float(os.getenv("EMBEDDER_POLL_SECONDS", "10"))
        if poll_seconds <= 0:
            raise RuntimeError("EMBEDDER_POLL_SECONDS must be positive")

        worker_name = os.getenv("EMBEDDER_WORKER_NAME", "embedder-1").strip()
        if not worker_name:
            raise RuntimeError("EMBEDDER_WORKER_NAME must not be empty")

        return cls(
            postgres_dsn=required("POSTGRES_DSN"),
            model_name=required("EMBEDDING_MODEL"),
            window_seconds=int(required("CHUNK_WINDOW_SECONDS")),
            max_records_per_chunk=int(required("MAX_RECORDS_PER_CHUNK")),
            embed_batch_size=int(required("EMBED_BATCH_SIZE")),
            mode=mode,
            poll_seconds=poll_seconds,
            worker_name=worker_name,
        )


@dataclass(frozen=True)
class ChunkGroup:
    records: tuple[Any, ...]
    window_start: datetime
    window_end: datetime
    sub_index: int


@dataclass(frozen=True)
class PreparedChunk:
    chunk_type: str
    window_start: datetime
    window_end: datetime
    sub_index: int
    services: list[str]
    trace_ids: list[str]
    max_level: str
    content: str
    raw_log_ids: list[int]


@dataclass(frozen=True)
class DirtyWindow:
    trace_id: str | None
    window_start: datetime


def utc_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def fixed_window_start(timestamp: datetime, window_seconds: int) -> datetime:
    timestamp = utc_timestamp(timestamp)
    epoch_seconds = int(timestamp.timestamp())
    grid_seconds = epoch_seconds - (epoch_seconds % window_seconds)
    return datetime.fromtimestamp(grid_seconds, tz=timezone.utc)


def group_records(
    records: Sequence[Any],
    *,
    window_seconds: int,
    max_records_per_chunk: int,
) -> list[ChunkGroup]:
    if max_records_per_chunk <= 0:
        raise ValueError("max_records_per_chunk must be positive")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    partitions: dict[str | None, list[Any]] = {}
    for record in records:
        partitions.setdefault(record["trace_id"], []).append(record)

    groups: list[ChunkGroup] = []
    for trace_id in sorted(
        partitions,
        key=lambda value: (value is not None, value or ""),
    ):
        groups.extend(
            _group_partition_records(
                partitions[trace_id],
                window_seconds=window_seconds,
                max_records_per_chunk=max_records_per_chunk,
            )
        )

    return groups


def _group_partition_records(
    records: Sequence[Any],
    *,
    window_seconds: int,
    max_records_per_chunk: int,
) -> list[ChunkGroup]:
    windows: dict[datetime, list[Any]] = {}
    for record in records:
        start = fixed_window_start(record["timestamp"], window_seconds)
        windows.setdefault(start, []).append(record)

    groups: list[ChunkGroup] = []
    for window_start in sorted(windows):
        window_records = sorted(
            windows[window_start],
            key=lambda record: (utc_timestamp(record["timestamp"]), record["id"]),
        )
        window_end = window_start + timedelta(seconds=window_seconds)

        if len(window_records) <= max_records_per_chunk:
            groups.append(
                ChunkGroup(tuple(window_records), window_start, window_end, sub_index=0)
            )
            continue

        for sub_index, offset in enumerate(
            range(0, len(window_records), max_records_per_chunk)
        ):
            sub_records = tuple(window_records[offset : offset + max_records_per_chunk])
            sub_start = min(utc_timestamp(record["timestamp"]) for record in sub_records)
            sub_end = max(utc_timestamp(record["timestamp"]) for record in sub_records)
            groups.append(ChunkGroup(sub_records, sub_start, sub_end, sub_index))

    return groups


def format_content(records: Iterable[Any]) -> str:
    lines = []
    for record in records:
        timestamp = utc_timestamp(record["timestamp"])
        lines.append(
            f"[{timestamp:%H:%M:%S}] {record['level']} "
            f"{record['service']}: {record['message']}"
        )
    return "\n".join(lines)


def collect_trace_ids(records: Iterable[Any]) -> list[str]:
    values = [record["trace_id"] for record in records]
    return sorted({value for value in values if value is not None})


LEVEL_RANK = {
    "DEBUG": 0,
    "INFO": 1,
    "WARN": 2,
    "ERROR": 3,
    "CRITICAL": 4,
}


def collect_max_level(records: Iterable[Any]) -> str:
    levels = [str(record["level"]).upper() for record in records]
    if not levels:
        raise ValueError("a chunk must contain at least one log record")
    unknown = [level for level in levels if level not in LEVEL_RANK]
    if unknown:
        raise ValueError(f"unknown log level(s): {', '.join(sorted(set(unknown)))}")
    return max(levels, key=LEVEL_RANK.__getitem__)


def prepare_chunk(group: ChunkGroup) -> PreparedChunk:
    records = group.records
    return PreparedChunk(
        chunk_type="window",
        window_start=group.window_start,
        window_end=group.window_end,
        sub_index=group.sub_index,
        services=sorted({record["service"] for record in records}),
        trace_ids=collect_trace_ids(records),
        max_level=collect_max_level(records),
        content=format_content(records),
        raw_log_ids=[record["id"] for record in records],
    )


def load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embedding_literal(vector: Any) -> str:
    values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    if len(values) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"expected {EMBEDDING_DIMENSION}-dimensional embedding, got {len(values)}"
        )
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def batched(items: Sequence[PreparedChunk], size: int) -> Iterable[Sequence[PreparedChunk]]:
    if size <= 0:
        raise ValueError("embed_batch_size must be positive")
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def embed_chunks(model: Any, chunks: Sequence[PreparedChunk], batch_size: int) -> list[str]:
    embeddings: list[str] = []
    for batch in batched(chunks, batch_size):
        vectors = model.encode(
            [chunk.content for chunk in batch],
            batch_size=len(batch),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if len(vectors) != len(batch):
            raise ValueError("embedding model returned an unexpected batch size")
        embeddings.extend(embedding_literal(vector) for vector in vectors)
    return embeddings


def load_table_schema(path: Path = SCHEMA_FILE) -> str:
    return path.read_text(encoding="utf-8")


async def ensure_database_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection:
        await connection.execute(load_table_schema())


async def load_raw_logs(pool: asyncpg.Pool) -> list[asyncpg.Record]:
    async with pool.acquire() as connection:
        return list(await connection.fetch(RAW_LOGS_SQL))


async def load_raw_logs_after(
    pool: asyncpg.Pool,
    last_raw_log_id: int,
) -> list[asyncpg.Record]:
    async with pool.acquire() as connection:
        return list(await connection.fetch(RAW_LOGS_AFTER_SQL, last_raw_log_id))


async def load_raw_logs_for_window(
    pool: asyncpg.Pool,
    dirty_window: DirtyWindow,
    window_seconds: int,
) -> list[asyncpg.Record]:
    window_start = utc_timestamp(dirty_window.window_start)
    window_end = window_start + timedelta(seconds=window_seconds)
    async with pool.acquire() as connection:
        return list(
            await connection.fetch(
                RAW_LOGS_WINDOW_SQL,
                dirty_window.trace_id,
                window_start,
                window_end,
            )
        )


def dirty_windows(
    records: Iterable[Any],
    *,
    window_seconds: int,
) -> list[DirtyWindow]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    windows = {
        DirtyWindow(
            trace_id=record["trace_id"],
            window_start=fixed_window_start(record["timestamp"], window_seconds),
        )
        for record in records
    }
    return sorted(
        windows,
        key=lambda value: (
            value.window_start,
            value.trace_id is not None,
            value.trace_id or "",
        ),
    )


async def load_checkpoint(pool: asyncpg.Pool, worker_name: str) -> int:
    async with pool.acquire() as connection:
        value = await connection.fetchval(CHECKPOINT_SQL, worker_name)
    return int(value or 0)


async def save_checkpoint(
    pool: asyncpg.Pool,
    worker_name: str,
    last_raw_log_id: int,
) -> None:
    async with pool.acquire() as connection:
        await connection.execute(UPSERT_CHECKPOINT_SQL, worker_name, last_raw_log_id)


async def insert_chunks(
    connection: asyncpg.Connection,
    chunks: Sequence[PreparedChunk],
    embeddings: Sequence[str],
    batch_size: int,
) -> None:
    for offset in range(0, len(chunks), batch_size):
        chunk_batch = chunks[offset : offset + batch_size]
        embedding_batch = embeddings[offset : offset + batch_size]
        await connection.executemany(
            INSERT_CHUNK_SQL,
            [
                (
                    chunk.chunk_type,
                    chunk.window_start,
                    chunk.window_end,
                    chunk.sub_index,
                    chunk.services,
                    chunk.trace_ids,
                    chunk.max_level,
                    chunk.content,
                    embedding,
                    chunk.raw_log_ids,
                )
                for chunk, embedding in zip(chunk_batch, embedding_batch)
            ],
        )


async def replace_chunks(
    pool: asyncpg.Pool,
    chunks: Sequence[PreparedChunk],
    embeddings: Sequence[str],
    batch_size: int,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute("TRUNCATE TABLE chunks")
            await insert_chunks(connection, chunks, embeddings, batch_size)


async def replace_dirty_window(
    pool: asyncpg.Pool,
    model: Any,
    config: EmbedderConfig,
    dirty_window: DirtyWindow,
) -> tuple[int, int]:
    records = await load_raw_logs_for_window(
        pool,
        dirty_window,
        config.window_seconds,
    )
    if not records:
        return 0, 0

    chunks = [
        prepare_chunk(group)
        for group in group_records(
            records,
            window_seconds=config.window_seconds,
            max_records_per_chunk=config.max_records_per_chunk,
        )
    ]
    embeddings = embed_chunks(model, chunks, config.embed_batch_size)
    raw_log_ids = [record["id"] for record in records]

    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                "DELETE FROM chunks WHERE raw_log_ids && $1::bigint[]",
                raw_log_ids,
            )
            await insert_chunks(connection, chunks, embeddings, config.embed_batch_size)

    return len(records), len(chunks)


async def run(config: EmbedderConfig | None = None) -> int:
    config = config or EmbedderConfig.from_env()
    pool = await asyncpg.create_pool(config.postgres_dsn)
    try:
        await ensure_database_schema(pool)
        raw_logs = await load_raw_logs(pool)
        chunks = [
            prepare_chunk(group)
            for group in group_records(
                raw_logs,
                window_seconds=config.window_seconds,
                max_records_per_chunk=config.max_records_per_chunk,
            )
        ]

        if chunks:
            logger.info("loading embedding model %s", config.model_name)
            model = load_embedding_model(config.model_name)
            embeddings = embed_chunks(model, chunks, config.embed_batch_size)
        else:
            embeddings = []

        await replace_chunks(pool, chunks, embeddings, config.embed_batch_size)
        logger.info("rebuilt %d chunks from %d raw logs", len(chunks), len(raw_logs))
        return len(chunks)
    finally:
        await pool.close()


async def run_live(config: EmbedderConfig | None = None) -> None:
    """Continuously embed only trace/window groups affected by new raw logs."""

    config = config or EmbedderConfig.from_env()
    pool = await asyncpg.create_pool(config.postgres_dsn)
    model = None
    try:
        await ensure_database_schema(pool)
        logger.info(
            "live embedder started worker=%s poll_seconds=%s",
            config.worker_name,
            config.poll_seconds,
        )
        while True:
            last_raw_log_id = await load_checkpoint(pool, config.worker_name)
            new_records = await load_raw_logs_after(pool, last_raw_log_id)
            if new_records:
                if model is None:
                    logger.info("loading embedding model %s", config.model_name)
                    model = load_embedding_model(config.model_name)

                windows = dirty_windows(
                    new_records,
                    window_seconds=config.window_seconds,
                )
                total_records = 0
                total_chunks = 0
                for window in windows:
                    records_count, chunks_count = await replace_dirty_window(
                        pool,
                        model,
                        config,
                        window,
                    )
                    total_records += records_count
                    total_chunks += chunks_count

                newest_id = max(record["id"] for record in new_records)
                await save_checkpoint(pool, config.worker_name, newest_id)
                logger.info(
                    "embedded %d new raw logs across %d dirty windows; refreshed %d chunks; cursor=%d",
                    len(new_records),
                    len(windows),
                    total_chunks,
                    newest_id,
                )

            await asyncio.sleep(config.poll_seconds)
    finally:
        await pool.close()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = EmbedderConfig.from_env()
    if config.mode == "live":
        asyncio.run(run_live(config))
    else:
        asyncio.run(run(config))


if __name__ == "__main__":
    main()
