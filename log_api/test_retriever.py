from datetime import datetime, timezone
from uuid import uuid4

from log_api.query_understanding import parse_query
from log_api.retriever import (
    RetrievedChunk,
    _filtered_dense_sql,
    _trace_sql,
    choose_trace_expansion,
    reciprocal_rank_fusion,
    select_trace_ids_for_expansion,
)


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


def test_filtered_dense_sql_has_service_and_time_predicates_without_severity_filter():
    parsed = parse_query(
        "postgres errors last night",
        datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
    )

    sql, values = _filtered_dense_sql(parsed)

    assert "services && $3::text[]" in sql
    assert "max_level = ANY" not in sql
    assert "window_start < $4::timestamptz" in sql
    assert "window_end > $5::timestamptz" in sql
    assert values == [
        ["postgres"],
        datetime(2026, 1, 1, 6, tzinfo=timezone.utc),
        datetime(2025, 12, 31, 18, tzinfo=timezone.utc),
    ]


def test_trace_expansion_removes_service_and_time_boundaries():
    sql, values = _trace_sql(max_chunks=50)

    assert "ANY(trace_ids)" in sql
    assert "services &&" not in sql
    assert "max_level = ANY" not in sql
    assert "window_start <" not in sql
    assert "window_end >" not in sql
    assert values == [25, 25]


def test_trace_sql_can_bound_database_fetch_to_trace_head_and_tail():
    sql, values = _trace_sql(max_chunks=5)

    assert "row_number() OVER" in sql
    assert "first_position <= $3" in sql
    assert "last_position <= $4" in sql
    assert values == [3, 2]


def test_empty_trace_expansion_retains_initial_filtered_results():
    initial = [make_chunk("postgres symptom")]

    assert choose_trace_expansion(initial, []) == initial


def test_close_trace_scores_expand_both_traces():
    first = make_chunk("first").model_copy(
        update={"trace_ids": ["trace-a"], "rerank_score": 10.0}
    )
    second = make_chunk("second").model_copy(
        update={"trace_ids": ["trace-b"], "rerank_score": 9.5}
    )

    selection = select_trace_ids_for_expansion([first, second], score_gap=1.0)

    assert selection.trace_ids == ["trace-a", "trace-b"]
    assert selection.note is not None


def test_distant_second_trace_is_not_expanded():
    first = make_chunk("first").model_copy(
        update={"trace_ids": ["trace-a"], "rerank_score": 10.0}
    )
    second = make_chunk("second").model_copy(
        update={"trace_ids": ["trace-b"], "rerank_score": 7.0}
    )

    selection = select_trace_ids_for_expansion([first, second], score_gap=1.0)

    assert selection.trace_ids == ["trace-a"]
    assert selection.note is None


def test_dense_cosine_threshold_can_keep_a_meaningfully_weaker_trace_out():
    first = make_chunk("first").model_copy(
        update={"trace_ids": ["trace-a"], "cosine_similarity": 0.80}
    )
    second = make_chunk("second").model_copy(
        update={"trace_ids": ["trace-b"], "cosine_similarity": 0.74}
    )

    selection = select_trace_ids_for_expansion([first, second], score_gap=0.05)

    assert selection.trace_ids == ["trace-a"]


def test_three_traces_expand_top_two_and_note_additional_trace():
    chunks = [
        make_chunk("first").model_copy(
            update={"trace_ids": ["trace-a"], "rerank_score": 10.0}
        ),
        make_chunk("second").model_copy(
            update={"trace_ids": ["trace-b"], "rerank_score": 9.0}
        ),
        make_chunk("third").model_copy(
            update={"trace_ids": ["trace-c"], "rerank_score": 8.0}
        ),
    ]

    selection = select_trace_ids_for_expansion(chunks, score_gap=1.0)

    assert selection.trace_ids == ["trace-a", "trace-b"]
    assert selection.note is not None
