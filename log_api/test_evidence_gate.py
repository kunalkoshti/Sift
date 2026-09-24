import pytest
from datetime import datetime, timezone
from uuid import UUID

from log_api.evidence_gate import (
    parse_evidence_verification,
    validate_evidence_verification,
)
from log_api.retriever import RetrievedChunk


def _chunk() -> RetrievedChunk:
    timestamp = datetime(2026, 1, 1, 23, 59, tzinfo=timezone.utc)
    return RetrievedChunk(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        window_start=timestamp,
        window_end=timestamp,
        sub_index=0,
        services=["payment-service"],
        trace_ids=["trace-payment"],
        content="[23:59:00] ERROR payment-service: Payment authorization timed out",
        cosine_distance=0.1,
        cosine_similarity=0.9,
    )


def test_parse_valid_evidence_verification():
    result = parse_evidence_verification(
        '{"supported": true, '
        '"unsupported_claims": [], '
        '"evidence_references": [{'
        '"claim": "The worker was killed by OOM", '
        '"chunk_id": "chunk-1", '
        '"timestamp": "23:59:00", '
        '"service": "payment-service"}], '
        '"reason": "The claims are supported."}'
    )

    assert result.supported
    assert result.unsupported_claim_count == 0
    assert result.evidence_references[0].service == "payment-service"
    assert not result.requires_action(0)


def test_parse_allows_json_inside_markdown_fence():
    result = parse_evidence_verification(
        '```json\n'
        '{"supported": false, "unsupported_claims": ["An invented event"], '
        '"evidence_references": [], "reason": "Not in context."}\n'
        '```'
    )

    assert not result.supported
    assert result.unsupported_claims == ("An invented event",)
    assert result.requires_action(0)


def test_supported_false_requires_action_even_without_claim_list():
    result = parse_evidence_verification(
        '{"supported": false, "unsupported_claims": [], '
        '"evidence_references": [], "reason": "The answer is not grounded."}'
    )

    assert result.requires_action(0)


@pytest.mark.parametrize(
    "payload",
    [
        '{"supported": "yes", "unsupported_claims": [], "evidence_references": []}',
        '{"supported": true, "unsupported_claims": [1], "evidence_references": []}',
        '{"supported": true, "unsupported_claims": [], "evidence_references": [{"claim": "x"}]}',
        "not json",
    ],
)
def test_invalid_verifier_output_is_rejected(payload):
    with pytest.raises(ValueError):
        parse_evidence_verification(payload)


def test_reference_must_point_to_real_chunk_timestamp_and_service():
    verification = parse_evidence_verification(
        '{"supported": true, "unsupported_claims": [], '
        '"evidence_references": [{'
        '"claim": "Authorization timed out", '
        '"chunk_id": "00000000-0000-0000-0000-000000000001", '
        '"timestamp": "23:59:00", '
        '"service": "payment-service"}], '
        '"reason": "Supported."}'
    )

    result = validate_evidence_verification(verification, [_chunk()], "The authorization timed out.")

    assert result.supported
    assert not result.requires_action(0)


@pytest.mark.parametrize(
    "reference",
    [
        '"chunk_id": "00000000-0000-0000-0000-000000000099", "timestamp": "23:59:00", "service": "payment-service"',
        '"chunk_id": "00000000-0000-0000-0000-000000000001", "timestamp": "23:58:00", "service": "payment-service"',
        '"chunk_id": "00000000-0000-0000-0000-000000000001", "timestamp": "23:59:00", "service": "checkout-service"',
    ],
)
def test_invalid_reference_requires_action(reference):
    verification = parse_evidence_verification(
        '{"supported": true, "unsupported_claims": [], '
        '"evidence_references": [{'
        f'"claim": "Authorization timed out", {reference}'
        '}], '
        '"reason": "Supported."}'
    )

    result = validate_evidence_verification(verification, [_chunk()], "The authorization timed out.")

    assert not result.supported
    assert result.requires_action(0)


def test_supported_answer_without_references_requires_action():
    verification = parse_evidence_verification(
        '{"supported": true, "unsupported_claims": [], '
        '"evidence_references": [], "reason": "Supported."}'
    )

    result = validate_evidence_verification(verification, [_chunk()], "The authorization timed out.")

    assert not result.supported
    assert result.requires_action(0)
