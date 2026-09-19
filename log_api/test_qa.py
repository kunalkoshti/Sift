from datetime import datetime, timezone
from uuid import UUID

from log_api.qa import (
    SYSTEM_PROMPT,
    deterministic_empty_context_answer,
    format_context,
)
from log_api.retriever import RetrievedChunk


def _chunk(minute: int) -> RetrievedChunk:
    start = datetime(2026, 1, 1, 0, minute, tzinfo=timezone.utc)
    return RetrievedChunk(
        id=UUID(f"00000000-0000-0000-0000-{minute + 1:012d}"),
        window_start=start,
        window_end=start,
        sub_index=0,
        services=["payment-service"],
        trace_ids=["trace-payment"],
        content=f"[{start:%H:%M:%S}] ERROR payment-service: event-{minute}",
        cosine_distance=0.1,
        cosine_similarity=0.9,
    )


def test_format_context_always_sorts_chunks_chronologically():
    context = format_context([_chunk(2), _chunk(0), _chunk(1)])

    assert context.index("event-0") < context.index("event-1")
    assert context.index("event-1") < context.index("event-2")


def test_prompt_separates_multiple_traces():
    assert "analyze each trace separately" in SYSTEM_PROMPT
    assert "do not\nmerge events from different traces" in SYSTEM_PROMPT
    assert "preserve that\nambiguity" in SYSTEM_PROMPT
    assert "absence of another trace" in SYSTEM_PROMPT
    assert "retrieved context" in SYSTEM_PROMPT


def test_prompt_requires_fallback_disclosure():
    assert "requested time range and" in SYSTEM_PROMPT


def test_format_context_has_a_hard_character_limit():
    context = format_context([_chunk(0), _chunk(1), _chunk(2)], max_chars=180)

    assert len(context) <= 180
    assert "CONTEXT TRUNCATED" in context


def test_empty_context_answer_is_deterministic():
    assert deterministic_empty_context_answer() == (
        "The available logs contain no evidence that can answer this question."
    )
    assert "No trace-correlated incident" in deterministic_empty_context_answer(
        "No chunks were found for the requested service(s): inventory-service."
    )
