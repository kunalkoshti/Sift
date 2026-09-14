"""Dense, PostgreSQL full-text, and reranked log retrieval."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import logging
from typing import Any, Sequence
from uuid import UUID

import asyncpg
from pydantic import BaseModel, ConfigDict

from log_api.query_understanding import (
    DEFAULT_SERVICE_CATALOG,
    ParsedQuery,
    parse_query,
)


logger = logging.getLogger(__name__)


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
  max_level,
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
  max_level,
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
  max_level,
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
    max_level: str | None = None
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
    trace_candidate_k: int = 20
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = DEFAULT_RRF_K
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    trace_rerank_score_gap: float = 1.0
    trace_cosine_score_gap: float = 0.05
    temporal_top_k: int = 10
    max_trace_chunks: int = 50


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    note: str | None = None


@dataclass(frozen=True)
class TraceSelection:
    trace_ids: list[str]
    note: str | None = None


def choose_trace_expansion(
    initial_chunks: list[RetrievedChunk],
    expanded_chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Use full-trace context when available, otherwise retain initial results."""

    return expanded_chunks if expanded_chunks else initial_chunks


def _chunk_ranking_score(chunk: RetrievedChunk) -> float:
    """Return a higher-is-better score for comparing initial candidates."""

    if chunk.rerank_score is not None:
        return float(chunk.rerank_score)
    if chunk.rrf_score is not None:
        return float(chunk.rrf_score)
    return float(chunk.cosine_similarity)


def select_trace_ids_for_expansion(
    chunks: Sequence[RetrievedChunk],
    score_gap: float,
) -> TraceSelection:
    """Select one or more incident traces from ranked initial candidates.

    ``score_gap`` is measured in the final candidate score: reranker score in
    hybrid mode, RRF score when available, or cosine similarity otherwise.
    """

    if score_gap < 0:
        raise ValueError("trace score gap must be non-negative")

    trace_scores: dict[str, float] = {}
    for chunk in chunks:
        score = _chunk_ranking_score(chunk)
        for trace_id in chunk.trace_ids:
            trace_scores[trace_id] = max(trace_scores.get(trace_id, score), score)

    ranked = sorted(trace_scores.items(), key=lambda item: (-item[1], item[0]))
    logger.debug("trace scores=%s; score gap=%s", ranked, score_gap)
    if not ranked:
        return TraceSelection(trace_ids=[])
    if len(ranked) == 1:
        return TraceSelection(trace_ids=[ranked[0][0]])

    best_score = ranked[0][1]
    second_score = ranked[1][1]
    if len(ranked) == 2:
        if best_score - second_score <= score_gap:
            return TraceSelection(
                trace_ids=[ranked[0][0], ranked[1][0]],
                note=(
                    "Two incident traces had similarly strong retrieval scores; "
                    "both traces are included for comparison."
                ),
            )
        return TraceSelection(trace_ids=[ranked[0][0]])

    selected = [ranked[0][0], ranked[1][0]]
    additional = len(ranked) - len(selected)
    return TraceSelection(
        trace_ids=selected,
        note=(
            f"{len(ranked)} incident traces were detected; the top two are "
            f"included and {additional} additional trace(s) were not expanded."
        ),
    )


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


def _filter_parts(
    parsed: ParsedQuery,
    start_parameter: int,
    *,
    include_time: bool,
) -> tuple[list[str], list[Any], int]:
    clauses: list[str] = []
    values: list[Any] = []
    parameter = start_parameter

    if parsed.service_filter:
        clauses.append(f"services && ${parameter}::text[]")
        values.append(parsed.service_filter)
        parameter += 1
    if include_time and parsed.time_range is not None:
        range_start, range_end = parsed.time_range
        clauses.append(
            f"window_start < ${parameter}::timestamptz "
            f"AND window_end > ${parameter + 1}::timestamptz"
        )
        # The overlap predicate uses end first and start second.
        values.extend([range_end, range_start])
        parameter += 2
    return clauses, values, parameter


def _filtered_dense_sql(parsed: ParsedQuery) -> tuple[str, list[Any]]:
    clauses, values, _ = _filter_parts(parsed, 3, include_time=True)
    where = "\n  AND " + "\n  AND ".join(clauses) if clauses else ""
    return (
        DENSE_RETRIEVE_SQL.replace(
            "WHERE embedding IS NOT NULL",
            "WHERE embedding IS NOT NULL" + where,
        ),
        values,
    )


def _filtered_lexical_sql(parsed: ParsedQuery) -> tuple[str, list[Any]]:
    clauses, values, _ = _filter_parts(parsed, 3, include_time=True)
    where = "\n  AND " + "\n  AND ".join(clauses) if clauses else ""
    return (
        LEXICAL_RETRIEVE_SQL.replace(
            "WHERE fts @@ websearch_to_tsquery('english', $1)",
            "WHERE fts @@ websearch_to_tsquery('english', $1)" + where,
        ),
        values,
    )


TEMPORAL_RETRIEVE_SQL = """
SELECT
  id,
  window_start,
  window_end,
  sub_index,
  services,
  trace_ids,
  max_level,
  content,
  NULL::double precision AS cosine_distance,
  NULL::double precision AS bm25_score
FROM chunks
WHERE window_start < $1::timestamptz
  AND window_end > $2::timestamptz
ORDER BY CASE max_level
  WHEN 'CRITICAL' THEN 4
  WHEN 'ERROR' THEN 3
  WHEN 'WARN' THEN 2
  WHEN 'INFO' THEN 1
  WHEN 'DEBUG' THEN 0
  ELSE -1
END DESC, window_start, sub_index, id
LIMIT $3
"""


def _trace_sql(max_chunks: int) -> tuple[str, list[Any]]:
    """Build the unfiltered, capped query used for trace expansion."""

    sql = TRACE_RETRIEVE_SQL
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")

    head_count = (max_chunks + 1) // 2
    tail_count = max_chunks - head_count
    # The base query is chronological. Numbering both directions lets the DB
    # bound the fetch while retaining the beginning and end of the trace.
    base_sql = sql.replace("ORDER BY window_start, sub_index, id", "")
    capped_sql = f"""
WITH ranked_trace AS (
  SELECT base_query.*,
         row_number() OVER (
           ORDER BY window_start, sub_index, id
         ) AS first_position,
         row_number() OVER (
           ORDER BY window_start DESC, sub_index DESC, id DESC
         ) AS last_position
  FROM ({base_sql}) AS base_query
)
SELECT id, window_start, window_end, sub_index, services, trace_ids,
       max_level, content, cosine_distance, bm25_score
FROM ranked_trace
WHERE first_position <= $3
   OR last_position <= $4
ORDER BY window_start, sub_index, id
"""
    return capped_sql, [head_count, tail_count]


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
        self.corpus_time_anchor: datetime | None = None

    async def start(self) -> None:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        if self.config.retrieval_mode not in {"dense", "hybrid"}:
            raise RuntimeError("RETRIEVAL_MODE must be either 'dense' or 'hybrid'")
        if self.config.top_k <= 0 or self.config.candidate_k <= 0:
            raise RuntimeError("retrieval limits must be positive")
        if self.config.trace_candidate_k <= 0:
            raise RuntimeError("trace_candidate_k must be positive")
        if self.config.temporal_top_k <= 0:
            raise RuntimeError("temporal_top_k must be positive")
        if self.config.max_trace_chunks <= 0:
            raise RuntimeError("max_trace_chunks must be positive")

        self.pool = await asyncpg.create_pool(self.config.postgres_dsn)
        async with self.pool.acquire() as connection:
            anchor = await connection.fetchval("SELECT MAX(window_end) FROM chunks")
        self.corpus_time_anchor = (
            anchor.astimezone(timezone.utc)
            if anchor is not None
            else datetime.now(timezone.utc)
        )
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
        self.corpus_time_anchor = None

    async def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Backward-compatible retrieval method returning only chunks."""

        result = await self.retrieve_with_context(question, top_k=top_k)
        return result.chunks

    async def retrieve_with_context(
        self,
        question: str,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Retrieve with deterministic query filters and an optional fallback note."""

        if self.pool is None or self.model is None:
            raise RuntimeError("retriever has not been started")

        limit = top_k if top_k is not None else self.config.top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")
        candidate_limit = max(limit, self.config.trace_candidate_k)

        anchor = self.corpus_time_anchor or datetime.now(timezone.utc)
        parsed = parse_query(question, anchor, DEFAULT_SERVICE_CATALOG)
        if parsed.unsupported_service:
            return RetrievalResult(
                chunks=[],
                note=(
                    f"The question names unrecognized service "
                    f"{parsed.unsupported_service!r}; no log retrieval was performed."
                ),
            )
        if parsed.unsupported_time_expression:
            return RetrievalResult(
                chunks=[],
                note=(
                    f"The temporal expression "
                    f"{parsed.unsupported_time_expression!r} is not supported; "
                    "no time-filtered retrieval was performed."
                ),
            )
        query_vector: str | None = None
        if parsed.intent != "temporal_only":
            query_vector = await asyncio.to_thread(encode_query, self.model, question)

        note: str | None = None
        async with self.pool.acquire() as connection:
            if parsed.intent == "temporal_only":
                range_start, range_end = parsed.time_range or (anchor, anchor)
                candidates = chunks_from_rows(
                    await connection.fetch(
                        TEMPORAL_RETRIEVE_SQL,
                        range_end,
                        range_start,
                    self.config.temporal_top_k,
                    )
                )
                temporal_trace_ids = sorted(
                    {
                        trace_id
                        for chunk in candidates
                        for trace_id in chunk.trace_ids
                    }
                )
                if len(temporal_trace_ids) > 1:
                    note = (
                        f"{len(temporal_trace_ids)} incident traces were found in "
                        "the requested time range; they are shown separately."
                    )
                elif candidates and not temporal_trace_ids:
                    note = (
                        "The requested time range contained only background noise "
                        "and no correlated incident traces."
                    )
            else:
                candidates = await self._retrieve_topic(
                    connection,
                    question,
                    parsed,
                    query_vector,
                    candidate_limit,
                )

            if not candidates and parsed.time_range is not None:
                fallback_query = replace(parsed, time_range=None, intent="topic_only")
                if query_vector is None:
                    query_vector = await asyncio.to_thread(encode_query, self.model, question)
                candidates = await self._retrieve_topic(
                    connection,
                    question,
                    fallback_query,
                    query_vector,
                    candidate_limit,
                )
                note = (
                    "No chunks found in the requested time range. "
                    "Answering from the full available time range while preserving "
                    "the requested service filter."
                )

            if query_vector is not None:
                initial_candidates = candidates[:limit]
                score_gap = (
                    self.config.trace_cosine_score_gap
                    if self.config.retrieval_mode == "dense"
                    else self.config.trace_rerank_score_gap
                )
                selection = select_trace_ids_for_expansion(
                    candidates,
                    score_gap,
                )
                if selection.trace_ids:
                    # Service and time filters identify the entry point. They
                    # are not boundaries for causal trace context.
                    trace_sql, trace_cap_values = _trace_sql(
                        self.config.max_trace_chunks,
                    )
                    trace_chunks: list[RetrievedChunk] = []
                    for trace_id in selection.trace_ids:
                        trace_chunks.extend(
                            chunks_from_rows(
                                await connection.fetch(
                                    trace_sql,
                                    trace_id,
                                    query_vector,
                                    *trace_cap_values,
                                )
                            )
                        )

                    score_by_id = {chunk.id: chunk for chunk in candidates}
                    expanded_candidates_by_id: dict[UUID, RetrievedChunk] = {}
                    for trace_chunk in trace_chunks:
                        initial = score_by_id.get(trace_chunk.id, trace_chunk)
                        expanded_candidates_by_id[trace_chunk.id] = trace_chunk.model_copy(
                            update={
                                "bm25_score": initial.bm25_score,
                                "dense_rank": initial.dense_rank,
                                "bm25_rank": initial.bm25_rank,
                                "rrf_score": initial.rrf_score,
                                "rerank_score": initial.rerank_score,
                            }
                        )
                    # The SQL head/tail window is the single authoritative
                    # trace-size limit. Keeping the cap in one place avoids
                    # divergent behavior between SQL and Python.
                    candidates = choose_trace_expansion(
                        initial_candidates,
                        list(expanded_candidates_by_id.values()),
                    )
                    if selection.note:
                        note = f"{note} {selection.note}" if note else selection.note
                else:
                    candidates = initial_candidates

            if parsed.service_filter and not candidates:
                services = ", ".join(parsed.service_filter)
                service_note = (
                    "No chunks were found for the requested service(s): "
                    f"{services}."
                )
                note = f"{note} {service_note}" if note else service_note
            elif parsed.service_filter and not any(
                chunk.trace_ids for chunk in candidates
            ):
                services = ", ".join(parsed.service_filter)
                service_note = (
                    "Chunks were found for the requested service(s), but none "
                    "belonged to a correlated incident trace: "
                    f"{services}."
                )
                note = f"{note} {service_note}" if note else service_note

        return RetrievalResult(chunks=candidates, note=note)

    async def _retrieve_topic(
        self,
        connection: Any,
        question: str,
        parsed: ParsedQuery,
        query_vector: str | None,
        limit: int,
    ) -> list[RetrievedChunk]:
        if query_vector is None:
            raise RuntimeError("topic retrieval requires a query embedding")

        if self.config.retrieval_mode == "dense":
            dense_sql, dense_filter_values = _filtered_dense_sql(parsed)
            return chunks_from_rows(
                await connection.fetch(
                    dense_sql,
                    query_vector,
                    limit,
                    *dense_filter_values,
                )
            )

        dense_sql, dense_filter_values = _filtered_dense_sql(parsed)
        lexical_sql, lexical_filter_values = _filtered_lexical_sql(parsed)
        dense_chunks = chunks_from_rows(
            await connection.fetch(
                dense_sql,
                query_vector,
                max(self.config.candidate_k, limit),
                *dense_filter_values,
            )
        )
        lexical_chunks = chunks_from_rows(
            await connection.fetch(
                lexical_sql,
                question,
                max(self.config.candidate_k, limit),
                *lexical_filter_values,
            )
        )
        fused = reciprocal_rank_fusion(
            dense_chunks,
            lexical_chunks,
            self.config.rrf_k,
            self.config.dense_weight,
            self.config.lexical_weight,
        )
        return (await self._rerank(question, fused))[:limit]

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
        try:
            raw_max_level = row["max_level"]
        except (KeyError, IndexError):
            raw_max_level = None
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
                max_level=raw_max_level,
                bm25_score=None if raw_bm25 is None else float(raw_bm25),
            )
        )
    return chunks
