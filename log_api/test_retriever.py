from datetime import datetime, timezone
from uuid import uuid4

from log_api.retriever import RetrievedChunk, reciprocal_rank_fusion


def make_chunk(content: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=uuid4(),
        window_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        sub_index=0,
        services=["payment-service"],
        trace_ids=["trace-1"],
        content=content,
        cosine_distance=0.2,
        cosine_similarity=0.8,
    )


def test_rrf_promotes_a_result_shared_by_dense_and_lexical_search() -> None:
    dense_shared = make_chunk("shared dense and lexical result")
    dense_only = make_chunk("dense only result")
    lexical_only = make_chunk("lexical only result")

    fused = reciprocal_rank_fusion(
        [dense_shared, dense_only],
        [lexical_only, dense_shared.model_copy(update={"bm25_score": 0.9})],
    )

    assert fused[0].id == dense_shared.id
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_rank == 2
    assert fused[0].rrf_score is not None


def test_rrf_rejects_non_positive_constant() -> None:
    chunk = make_chunk("result")

    try:
        reciprocal_rank_fusion([chunk], [], rrf_k=0)
    except ValueError as exc:
        assert str(exc) == "rrf_k must be positive"
    else:
        raise AssertionError("expected reciprocal_rank_fusion to reject rrf_k=0")


def test_rrf_accepts_independent_dense_and_lexical_weights() -> None:
    dense_chunk = make_chunk("dense result")
    lexical_chunk = make_chunk("lexical result")

    fused = reciprocal_rank_fusion(
        [dense_chunk],
        [lexical_chunk],
        dense_weight=0.8,
        lexical_weight=0.2,
    )

    assert fused[0].id == dense_chunk.id


def test_rrf_rejects_two_zero_weights() -> None:
    chunk = make_chunk("result")

    try:
        reciprocal_rank_fusion([chunk], [], dense_weight=0, lexical_weight=0)
    except ValueError as exc:
        assert str(exc) == "at least one RRF weight must be positive"
    else:
        raise AssertionError("expected reciprocal_rank_fusion to reject zero weights")
