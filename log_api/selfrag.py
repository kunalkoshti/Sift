"""Pre-generation confidence gating for self-RAG experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


VALID_CONFIDENCE_GATE_MODES = {"off", "shadow", "enforce"}


@dataclass(frozen=True)
class ConfidenceGateDecision:
    """The observable result of the confidence gate."""

    mode: str
    best_dense_similarity: float | None
    best_trace_dense_similarity: float | None
    has_trace: bool
    has_strong_trace: bool
    eligible: bool
    would_abstain: bool
    reason: str

    @property
    def enforced_abstention(self) -> bool:
        return self.mode == "enforce" and self.would_abstain


def evaluate_confidence_gate(
    candidates: Sequence[Any],
    *,
    threshold: float,
    mode: str,
    retrieval_mode: str,
    semantic_retrieval: bool = True,
    service_filter_applied: bool = False,
    fallback_used: bool = False,
    trace_candidates: Sequence[Any] | None = None,
) -> ConfidenceGateDecision:
    """Evaluate weak evidence before trace expansion or LLM generation.

    The threshold applies only to dense cosine similarity. RRF and cross-encoder
    values are ranking scores with different, uncalibrated scales. In hybrid
    mode, lexical-only candidates are therefore excluded from the dense-score
    calculation. Queries that did not perform semantic retrieval bypass the
    gate because they have no comparable confidence value.
    """

    if mode not in VALID_CONFIDENCE_GATE_MODES:
        raise ValueError(
            "confidence gate mode must be one of: "
            + ", ".join(sorted(VALID_CONFIDENCE_GATE_MODES))
        )
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("confidence threshold must be between -1 and 1")

    trace_source = candidates if trace_candidates is None else trace_candidates
    trace_ids = {
        trace_id
        for candidate in trace_source
        for trace_id in getattr(candidate, "trace_ids", ())
        if trace_id
    }
    has_trace = bool(trace_ids)

    top_k_has_trace = any(
        getattr(candidate, "trace_ids", ()) for candidate in candidates
    )

    def has_dense_score(candidate: Any) -> bool:
        return retrieval_mode == "dense" or getattr(candidate, "dense_rank", None) is not None

    dense_scores = [
        float(candidate.cosine_similarity)
        for candidate in candidates
        if has_dense_score(candidate)
        and getattr(candidate, "cosine_similarity", None) is not None
    ]
    trace_dense_scores = [
        float(candidate.cosine_similarity)
        for candidate in trace_source
        if getattr(candidate, "trace_ids", ())
        and has_dense_score(candidate)
        and getattr(candidate, "cosine_similarity", None) is not None
    ]
    best_dense_similarity = max(dense_scores) if dense_scores else None
    best_trace_dense_similarity = (
        max(trace_dense_scores) if trace_dense_scores else None
    )
    has_strong_trace = top_k_has_trace or (
        best_trace_dense_similarity is not None
        and best_trace_dense_similarity >= threshold
    )

    if service_filter_applied:
        return ConfidenceGateDecision(
            mode=mode,
            best_dense_similarity=best_dense_similarity,
            best_trace_dense_similarity=best_trace_dense_similarity,
            has_trace=has_trace,
            has_strong_trace=has_strong_trace,
            eligible=False,
            would_abstain=False,
            reason="recognized_service_filter_returned_results",
        )

    if fallback_used:
        return ConfidenceGateDecision(
            mode=mode,
            best_dense_similarity=best_dense_similarity,
            best_trace_dense_similarity=best_trace_dense_similarity,
            has_trace=has_trace,
            has_strong_trace=has_strong_trace,
            eligible=False,
            would_abstain=False,
            reason="temporal_filter_fallback_used",
        )

    if not semantic_retrieval or mode == "off":
        return ConfidenceGateDecision(
            mode=mode,
            best_dense_similarity=best_dense_similarity,
            best_trace_dense_similarity=best_trace_dense_similarity,
            has_trace=has_trace,
            has_strong_trace=has_strong_trace,
            eligible=False,
            would_abstain=False,
            reason="gate_disabled_or_nonsemantic_query",
        )

    if not dense_scores:
        return ConfidenceGateDecision(
            mode=mode,
            best_dense_similarity=None,
            best_trace_dense_similarity=best_trace_dense_similarity,
            has_trace=has_trace,
            has_strong_trace=has_strong_trace,
            eligible=False,
            would_abstain=False,
            reason="no_dense_score_available",
        )

    would_abstain = best_dense_similarity < threshold and not has_strong_trace
    if would_abstain:
        reason = "best_dense_similarity_below_threshold_without_strong_trace"
    elif top_k_has_trace:
        reason = "trace_found_in_top_k"
    elif has_strong_trace:
        reason = "strong_trace_found_in_candidate_pool"
    elif has_trace:
        reason = "weak_trace_found_in_candidate_pool"
    else:
        reason = "dense_similarity_meets_threshold"

    return ConfidenceGateDecision(
        mode=mode,
        best_dense_similarity=best_dense_similarity,
        best_trace_dense_similarity=best_trace_dense_similarity,
        has_trace=has_trace,
        has_strong_trace=has_strong_trace,
        eligible=True,
        would_abstain=would_abstain,
        reason=reason,
    )
