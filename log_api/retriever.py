"""Dense, PostgreSQL full-text, and reranked log retrieval."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence
from uuid import UUID

import asyncpg
from pydantic import BaseModel, ConfigDict


EMBEDDING_DIMENSION = 384
DEFAULT_RRF_K = 60

DENSE_RETRIEVE_SQL = """
SELECT
  id,
  window_start,
  window_end,
  sub_index,
  services,
  trace_ids,
  content,
  embedding <=> $1::vector AS cosine_distance,
  NULL::double precision AS bm25_score
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT $2
"""

LEXICAL_RETRIEVE_SQL = """
SELECT
  id,
  window_start,
  window_end,
  sub_index,
  services,
  trace_ids,
  content,
  NULL::double precision AS cosine_distance,
  ts_rank_cd(fts, websearch_to_tsquery('english', $1)) AS bm25_score
FROM chunks
WHERE fts @@ websearch_to_tsquery('english', $1)
ORDER BY bm25_score DESC, window_start, sub_index, id
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
  embedding <=> $2::vector AS cosine_distance,
  NULL::double precision AS bm25_score
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
    bm25_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class RetrieverConfig:
    postgres_dsn: str
    embedding_model: str
    top_k: int
    retrieval_mode: str = "hybrid"
    candidate_k: int = 20
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = DEFAULT_RRF_K
    dense_weight: float = 1.0
    lexical_weight: float = 1.0


def vector_literal(vector: Any) -> str:
    values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    if len(values) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"expected {EMBEDDING_DIMENSION}-dimensional query embedding, got {len(values)}"
        )
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def encode_query(model: Any, question: str) -> str:
    """Encode exactly like log_embedder: BGE plus normalized embeddings."""

    vector = model.encode(
        [question],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return vector_literal(vector)


def reciprocal_rank_fusion(
    dense_chunks: Sequence[RetrievedChunk],
    lexical_chunks: Sequence[RetrievedChunk],
    rrf_k: int = DEFAULT_RRF_K,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> list[RetrievedChunk]:
    """Fuse dense and lexical rankings using weighted reciprocal rank fusion."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if dense_weight < 0 or lexical_weight < 0:
        raise ValueError("RRF weights must be non-negative")
    if dense_weight == 0 and lexical_weight == 0:
        raise ValueError("at least one RRF weight must be positive")

    merged: dict[UUID, RetrievedChunk] = {}
    rrf_scores: dict[UUID, float] = {}

    for rank, chunk in enumerate(dense_chunks, start=1):
        merged[chunk.id] = chunk.model_copy(update={"dense_rank": rank})
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (
            dense_weight / (rrf_k + rank)
        )

    for rank, chunk in enumerate(lexical_chunks, start=1):
        if chunk.id in merged:
            merged[chunk.id] = merged[chunk.id].model_copy(
                update={
                    "bm25_score": chunk.bm25_score,
                    "bm25_rank": rank,
                }
            )
        else:
            merged[chunk.id] = chunk.model_copy(update={"bm25_rank": rank})
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (
            lexical_weight / (rrf_k + rank)
        )

    return sorted(
        (
            chunk.model_copy(update={"rrf_score": rrf_scores[chunk.id]})
            for chunk in merged.values()
        ),
        key=lambda chunk: (-float(chunk.rrf_score or 0.0), str(chunk.id)),
    )


class HybridRetriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.pool: asyncpg.Pool | None = None
        self.model: Any | None = None
        self.reranker: Any | None = None

    async def start(self) -> None:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        if self.config.retrieval_mode not in {"dense", "hybrid"}:
            raise RuntimeError("RETRIEVAL_MODE must be either 'dense' or 'hybrid'")
        if self.config.top_k <= 0 or self.config.candidate_k <= 0:
            raise RuntimeError("retrieval limits must be positive")

        self.pool = await asyncpg.create_pool(self.config.postgres_dsn)
        self.model = await asyncio.to_thread(
            SentenceTransformer,
            self.config.embedding_model,
        )
        if self.config.retrieval_mode == "hybrid":
            self.reranker = await asyncio.to_thread(
                CrossEncoder,
                self.config.reranker_model,
            )

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
        self.model = None
        self.reranker = None

    async def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve candidates, optionally fuse and rerank, then expand one trace."""

        if self.pool is None or self.model is None:
            raise RuntimeError("retriever has not been started")

        limit = top_k if top_k is not None else self.config.top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")

        query_vector = await asyncio.to_thread(encode_query, self.model, question)
        async with self.pool.acquire() as connection:
            if self.config.retrieval_mode == "dense":
                candidates = chunks_from_rows(
                    await connection.fetch(DENSE_RETRIEVE_SQL, query_vector, limit)
                )
            else:
                dense_chunks = chunks_from_rows(
                    await connection.fetch(
                        DENSE_RETRIEVE_SQL,
                        query_vector,
                        self.config.candidate_k,
                    )
                )
                lexical_chunks = chunks_from_rows(
                    await connection.fetch(
                        LEXICAL_RETRIEVE_SQL,
                        question,
                        self.config.candidate_k,
                    )
                )
                fused = reciprocal_rank_fusion(
                    dense_chunks,
                    lexical_chunks,
                    self.config.rrf_k,
                    self.config.dense_weight,
                    self.config.lexical_weight,
                )
                candidates = await self._rerank(question, fused)
                candidates = candidates[:limit]

            if candidates and candidates[0].trace_ids:
                trace_chunks = chunks_from_rows(
                    await connection.fetch(
                        TRACE_RETRIEVE_SQL,
                        candidates[0].trace_ids[0],
                        query_vector,
                    )
                )
                score_by_id = {chunk.id: chunk for chunk in candidates}
                candidates = [
                    trace_chunk.model_copy(
                        update={
                            "bm25_score": score_by_id.get(trace_chunk.id, trace_chunk).bm25_score,
                            "dense_rank": score_by_id.get(trace_chunk.id, trace_chunk).dense_rank,
                            "bm25_rank": score_by_id.get(trace_chunk.id, trace_chunk).bm25_rank,
                            "rrf_score": score_by_id.get(trace_chunk.id, trace_chunk).rrf_score,
                            "rerank_score": score_by_id.get(trace_chunk.id, trace_chunk).rerank_score,
                        }
                    )
                    for trace_chunk in trace_chunks
                ]

        return candidates

    async def _rerank(
        self,
        question: str,
        candidates: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if self.reranker is None:
            raise RuntimeError("hybrid retriever reranker has not been started")
        if not candidates:
            return []

        pairs = [[question, chunk.content] for chunk in candidates]
        scores = await asyncio.to_thread(
            self.reranker.predict,
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )
        reranked = [
            chunk.model_copy(update={"rerank_score": float(score)})
            for chunk, score in zip(candidates, scores, strict=True)
        ]
        return sorted(
            reranked,
            key=lambda chunk: (-float(chunk.rerank_score or 0.0), str(chunk.id)),
        )


# Backward-compatible name for callers that still import the Stage 0 class.
DenseRetriever = HybridRetriever


def chunks_from_rows(rows: Sequence[Any]) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for row in rows:
        raw_distance = row["cosine_distance"]
        distance = 1.0 if raw_distance is None else float(raw_distance)
        raw_bm25 = row["bm25_score"]
        chunks.append(
            RetrievedChunk(
                id=row["id"],
                window_start=row["window_start"],
                window_end=row["window_end"],
                sub_index=row["sub_index"],
                services=list(row["services"]),
                trace_ids=list(row["trace_ids"]),
                content=row["content"],
                cosine_distance=distance,
                cosine_similarity=1.0 - distance,
                bm25_score=None if raw_bm25 is None else float(raw_bm25),
            )
        )
    return chunks
