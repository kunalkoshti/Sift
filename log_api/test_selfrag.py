from dataclasses import dataclass

import pytest

from log_api.selfrag import evaluate_confidence_gate


@dataclass
class Candidate:
    cosine_similarity: float | None
    trace_ids: list[str]
    dense_rank: int | None = None


def test_low_confidence_without_trace_would_abstain_in_shadow_mode():
    decision = evaluate_confidence_gate(
        [Candidate(cosine_similarity=0.42, trace_ids=[], dense_rank=1)],
        threshold=0.55,
        mode="shadow",
        retrieval_mode="hybrid",
    )

    assert decision.eligible
    assert decision.would_abstain
    assert not decision.enforced_abstention
    assert decision.best_dense_similarity == 0.42


def test_low_confidence_without_trace_is_enforced_as_abstention():
    decision = evaluate_confidence_gate(
        [Candidate(cosine_similarity=0.42, trace_ids=[], dense_rank=1)],
        threshold=0.55,
        mode="enforce",
        retrieval_mode="hybrid",
    )

    assert decision.would_abstain
    assert decision.enforced_abstention


def test_trace_evidence_prevents_confidence_abstention():
    decision = evaluate_confidence_gate(
        [Candidate(cosine_similarity=0.42, trace_ids=["trace-1"], dense_rank=1)],
        threshold=0.55,
        mode="enforce",
        retrieval_mode="hybrid",
    )

    assert decision.eligible
    assert decision.has_trace
    assert decision.has_strong_trace
    assert not decision.would_abstain


def test_service_filter_with_noise_results_bypasses_confidence_gate():
    decision = evaluate_confidence_gate(
        [Candidate(cosine_similarity=0.42, trace_ids=[], dense_rank=1)],
        threshold=0.55,
        mode="enforce",
        retrieval_mode="hybrid",
        service_filter_applied=True,
    )

    assert not decision.eligible
    assert not decision.would_abstain
    assert decision.reason == "recognized_service_filter_returned_results"


def test_temporal_fallback_bypasses_confidence_gate():
    decision = evaluate_confidence_gate(
        [Candidate(cosine_similarity=0.42, trace_ids=[], dense_rank=1)],
        threshold=0.55,
        mode="enforce",
        retrieval_mode="hybrid",
        fallback_used=True,
    )

    assert not decision.eligible
    assert not decision.would_abstain
    assert decision.reason == "temporal_filter_fallback_used"


def test_trace_in_broader_candidate_pool_prevents_abstention():
    decision = evaluate_confidence_gate(
        [Candidate(cosine_similarity=0.42, trace_ids=[], dense_rank=1)],
        trace_candidates=[
            Candidate(cosine_similarity=0.42, trace_ids=[], dense_rank=1),
            Candidate(cosine_similarity=0.60, trace_ids=["trace-deeper"], dense_rank=2),
        ],
        threshold=0.55,
        mode="enforce",
        retrieval_mode="hybrid",
    )

    assert decision.eligible
    assert decision.has_trace
    assert decision.has_strong_trace
    assert not decision.would_abstain
    assert decision.reason == "strong_trace_found_in_candidate_pool"


def test_weak_trace_in_broader_candidate_pool_does_not_prevent_abstention():
    decision = evaluate_confidence_gate(
        [Candidate(cosine_similarity=0.42, trace_ids=[], dense_rank=1)],
        trace_candidates=[
            Candidate(cosine_similarity=0.42, trace_ids=[], dense_rank=1),
            Candidate(cosine_similarity=0.49, trace_ids=["trace-deeper"], dense_rank=2),
        ],
        threshold=0.55,
        mode="enforce",
        retrieval_mode="hybrid",
    )

    assert decision.has_trace
    assert not decision.has_strong_trace
    assert decision.would_abstain
    assert decision.reason == (
        "best_dense_similarity_below_threshold_without_strong_trace"
    )


def test_threshold_boundary_is_not_below_threshold():
    decision = evaluate_confidence_gate(
        [Candidate(cosine_similarity=0.55, trace_ids=[])],
        threshold=0.55,
        mode="enforce",
        retrieval_mode="dense",
    )

    assert not decision.would_abstain


def test_hybrid_lexical_only_candidate_bypasses_dense_gate():
    decision = evaluate_confidence_gate(
        [Candidate(cosine_similarity=0.0, trace_ids=[], dense_rank=None)],
        threshold=0.55,
        mode="enforce",
        retrieval_mode="hybrid",
    )

    assert not decision.eligible
    assert not decision.would_abstain
    assert decision.reason == "no_dense_score_available"


def test_nonsemantic_query_bypasses_gate():
    decision = evaluate_confidence_gate(
        [Candidate(cosine_similarity=None, trace_ids=[])],
        threshold=0.55,
        mode="enforce",
        retrieval_mode="hybrid",
        semantic_retrieval=False,
    )

    assert not decision.eligible
    assert not decision.would_abstain


def test_off_mode_does_not_abstain():
    decision = evaluate_confidence_gate(
        [Candidate(cosine_similarity=0.1, trace_ids=[])],
        threshold=0.55,
        mode="off",
        retrieval_mode="dense",
    )

    assert not decision.would_abstain
    assert not decision.enforced_abstention


def test_invalid_gate_configuration_is_rejected():
    with pytest.raises(ValueError, match="confidence gate mode"):
        evaluate_confidence_gate(
            [],
            threshold=0.55,
            mode="invalid",
            retrieval_mode="hybrid",
        )

    with pytest.raises(ValueError, match="between -1 and 1"):
        evaluate_confidence_gate(
            [],
            threshold=1.1,
            mode="shadow",
            retrieval_mode="hybrid",
        )
