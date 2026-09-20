import pytest

from log_api.evidence_gate import parse_evidence_verification


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
