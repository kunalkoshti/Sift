import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID

from log_api.app import RAGService
from log_api.evidence_gate import EvidenceVerification
from log_api.retriever import RetrievedChunk, RetrievalResult


def _chunk() -> RetrievedChunk:
    timestamp = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    return RetrievedChunk(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        window_start=timestamp,
        window_end=timestamp,
        sub_index=0,
        services=["payment-service"],
        trace_ids=["trace-payment"],
        content="Payment authorization timed out for order order-84721",
        cosine_distance=0.1,
        cosine_similarity=0.9,
    )


def _verification(*, supported: bool) -> EvidenceVerification:
    return EvidenceVerification(
        supported=supported,
        unsupported_claims=("The customer segment was enterprise",)
        if not supported
        else (),
        evidence_references=(),
        reason="fabricated claim is not present in context"
        if not supported
        else "all claims are supported",
        raw_output="{}",
    )


class FakeRetriever:
    async def retrieve_with_context(self, question: str) -> RetrievalResult:
        return RetrievalResult(chunks=[_chunk()])


class FakeQAChain:
    def __init__(self):
        self.revise_calls = 0

    async def answer(self, question, chunks, retrieval_note=None):
        return "The payment failures affected enterprise customers."

    async def revise(
        self,
        question,
        previous_answer,
        chunks,
        retrieval_note=None,
        unsupported_claims=(),
    ):
        self.revise_calls += 1
        return "The payment authorization timed out for order order-84721."


class FakeEvidenceVerifier:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def verify(self, question, answer, chunks, retrieval_note=None):
        self.calls += 1
        return self.responses.pop(0)


def _service(verifier, qa_chain):
    service = object.__new__(RAGService)
    service.config = SimpleNamespace(
        evidence_gate_mode="enforce",
        evidence_max_unsupported_claims=0,
        evidence_max_retries=1,
    )
    service.retriever = FakeRetriever()
    service.qa_chain = qa_chain
    service.evidence_verifier = verifier
    return service


def test_enforce_retries_fabricated_answer_and_abstains_after_second_failure():
    qa_chain = FakeQAChain()
    verifier = FakeEvidenceVerifier(
        [_verification(supported=False), _verification(supported=False)]
    )

    response = asyncio.run(
        _service(verifier, qa_chain).ask("Which customer segment had payment failures?")
    )

    assert qa_chain.revise_calls == 1
    assert verifier.calls == 2
    assert response.answer == (
        "The retrieved logs did not support all claims needed for a reliable "
        "answer, so I cannot answer this safely from the available evidence."
    )


def test_enforce_returns_revised_answer_when_second_check_passes():
    qa_chain = FakeQAChain()
    verifier = FakeEvidenceVerifier(
        [_verification(supported=False), _verification(supported=True)]
    )

    response = asyncio.run(
        _service(verifier, qa_chain).ask("Which customer segment had payment failures?")
    )

    assert qa_chain.revise_calls == 1
    assert verifier.calls == 2
    assert response.answer == "The payment authorization timed out for order order-84721."
