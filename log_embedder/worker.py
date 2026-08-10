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

INSERT_CHUNK_SQL = """
INSERT INTO chunks (
  chunk_type,
  window_start,
  window_end,
  sub_index,
  services,
  trace_ids,
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
  $8::vector,
  $9::bigint[]
)
"""


@dataclass(frozen=True)
class EmbedderConfig:
    postgres_dsn: str
    model_name: str
    window_seconds: int
    max_records_per_chunk: int
    embed_batch_size: int

    @classmethod
    def from_env(cls) -> EmbedderConfig:
        def required(name: str) -> str:
            value = os.getenv(name)
            if not value:
                raise RuntimeError(f"{name} must be configured in .env or the environment")
            return value

        return cls(
            postgres_dsn=required("POSTGRES_DSN"),
            model_name=required("EMBEDDING_MODEL"),
            window_seconds=int(required("CHUNK_WINDOW_SECONDS")),
            max_records_per_chunk=int(required("MAX_RECORDS_PER_CHUNK")),
            embed_batch_size=int(required("EMBED_BATCH_SIZE")),
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
    content: str
    raw_log_ids: list[int]


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


def prepare_chunk(group: ChunkGroup) -> PreparedChunk:
    records = group.records
    return PreparedChunk(
        chunk_type="window",
        window_start=group.window_start,
        window_end=group.window_end,
        sub_index=group.sub_index,
        services=sorted({record["service"] for record in records}),
        trace_ids=collect_trace_ids(records),
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
                            chunk.content,
                            embedding,
                            chunk.raw_log_ids,
                        )
                        for chunk, embedding in zip(chunk_batch, embedding_batch)
                    ],
                )


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


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(run())


if __name__ == "__main__":
    main()
