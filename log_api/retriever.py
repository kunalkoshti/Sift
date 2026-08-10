"""Dense pgvector retrieval using the same BGE encoding settings as the embedder."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence
from uuid import UUID

import asyncpg
from pydantic import BaseModel, ConfigDict


EMBEDDING_DIMENSION = 384

RETRIEVE_SQL = """
SELECT
  id,
  window_start,
  window_end,
  sub_index,
  services,
  trace_ids,
  content,
  embedding <=> $1::vector AS cosine_distance
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT $2
"""

TRACE_RETRIEVE_SQL = """
SELECT
  id,
  window_start,
  window_end,
  sub_index,
  services,
  trace_ids,
  content,
  embedding <=> $2::vector AS cosine_distance
FROM chunks
WHERE $1 = ANY(trace_ids)
  AND embedding IS NOT NULL
ORDER BY window_start, sub_index, id
"""


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    window_start: datetime
    window_end: datetime
    sub_index: int
    services: list[str]
    trace_ids: list[str]
    content: str
    cosine_distance: float
    cosine_similarity: float


@dataclass(frozen=True)
class RetrieverConfig:
    postgres_dsn: str
    embedding_model: str
    top_k: int


def vector_literal(vector: Any) -> str:
    values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    if len(values) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"expected {EMBEDDING_DIMENSION}-dimensional query embedding, got {len(values)}"
        )
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def encode_query(model: Any, question: str) -> str:
    """Encode exactly like log_embedder: BGE + normalized embeddings."""

    vector = model.encode(
        [question],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return vector_literal(vector)


class DenseRetriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.pool: asyncpg.Pool | None = None
        self.model: Any | None = None

    async def start(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.pool = await asyncpg.create_pool(self.config.postgres_dsn)
        self.model = await asyncio.to_thread(
            SentenceTransformer,
            self.config.embedding_model,
        )

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve dense matches, then expand the top result's incident trace.

        If the top-ranked chunk has a trace ID, all chunks for that trace are
        returned chronologically. A noise chunk with no trace ID falls back to
        the original dense top-k results. This intentionally does not resolve
        close-similarity ties between multiple plausible incidents.
        """
        if self.pool is None or self.model is None:
            raise RuntimeError("dense retriever has not been started")

        query_vector = await asyncio.to_thread(encode_query, self.model, question)
        limit = top_k if top_k is not None else self.config.top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")

        async with self.pool.acquire() as connection:
            rows = await connection.fetch(RETRIEVE_SQL, query_vector, limit)
            chunks = chunks_from_rows(rows)

            if chunks and chunks[0].trace_ids:
                rows = await connection.fetch(
                    TRACE_RETRIEVE_SQL,
                    chunks[0].trace_ids[0],
                    query_vector,
                )
                chunks = chunks_from_rows(rows)

        return chunks


def chunks_from_rows(rows: Sequence[Any]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            id=row["id"],
            window_start=row["window_start"],
            window_end=row["window_end"],
            sub_index=row["sub_index"],
            services=list(row["services"]),
            trace_ids=list(row["trace_ids"]),
            content=row["content"],
            cosine_distance=float(row["cosine_distance"]),
            cosine_similarity=1.0 - float(row["cosine_distance"]),
        )
        for row in rows
    ]
