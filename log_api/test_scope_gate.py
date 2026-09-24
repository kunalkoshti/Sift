import pytest
import asyncio
from types import SimpleNamespace

from log_api.app import RAGService
from log_api.scope_gate import (
    deterministic_scope_answer,
    evaluate_scope_gate,
)


@pytest.mark.parametrize(
    ("question", "category"),
    [
        ("What will the weather be tomorrow?", "external_state"),
        ("What was the company's revenue last quarter?", "financial_metric"),
        (
            "Which customer segment was most affected by checkout failures?",
            "customer_demographic",
        ),
        (
            "What percentage of payment attempts failed last night?",
            "aggregate_business_statistic",
        ),
    ],
)
def test_clearly_out_of_scope_questions_are_detected(question, category):
    decision = evaluate_scope_gate(question, mode="enforce")

    assert not decision.in_scope
    assert decision.eligible
    assert decision.would_abstain
    assert decision.enforced_abstention
    assert decision.category == category


@pytest.mark.parametrize(
    "question",
    [
        "Why did payment authorization time out?",
        "Are there any notable incidents involving the inventory service?",
        "What happened around midnight?",
        "What was the payment failure rate-limit error?",
    ],
)
def test_log_questions_are_not_blocked_by_scope_gate(question):
    decision = evaluate_scope_gate(question, mode="enforce")

    assert decision.in_scope
    assert not decision.would_abstain


def test_shadow_mode_reports_scope_without_blocking():
    decision = evaluate_scope_gate(
        "What was the company's revenue last quarter?",
        mode="shadow",
    )

    assert not decision.in_scope
    assert decision.would_abstain
    assert not decision.enforced_abstention


def test_off_mode_bypasses_scope_rules():
    decision = evaluate_scope_gate(
        "What will the weather be tomorrow?",
        mode="off",
    )

    assert decision.in_scope
    assert not decision.eligible
    assert not decision.would_abstain


def test_deterministic_scope_answer_explains_boundary():
    decision = evaluate_scope_gate(
        "What was the company's revenue last quarter?",
        mode="enforce",
    )

    answer = deterministic_scope_answer(decision)

    assert "cannot answer" in answer
    assert "financial metrics" in answer


def test_invalid_scope_configuration_is_rejected():
    with pytest.raises(ValueError, match="scope gate mode"):
        evaluate_scope_gate("Why did payments fail?", mode="invalid")

    with pytest.raises(ValueError, match="must not be blank"):
        evaluate_scope_gate("   ", mode="shadow")


def test_enforced_scope_gate_returns_before_retrieval():
    class RetrievalMustNotRun:
        async def retrieve_with_context(self, question):
            raise AssertionError("retrieval should not run for an enforced scope rejection")

    service = object.__new__(RAGService)
    service.config = SimpleNamespace(scope_gate_mode="enforce")
    service.retriever = RetrievalMustNotRun()

    response = asyncio.run(
        service.ask("What was the company's revenue last quarter?")
    )

    assert response.retrieved_chunks == []
    assert response.retrieval_note == (
        "Scope gate classified the question as financial_metric."
    )
    assert "financial metrics" in response.answer
