import json

from evaluation.behavior_classifier import (
    VALID_BEHAVIORS,
    behavior_matches,
    classify_empty_retrieval,
    parse_classification,
)
from evaluation.provider import EvaluationLLMConfig
from evaluation.ragas_metrics import should_score_context_metrics


def test_question_file_is_structured_and_covers_all_scenarios():
    with open("evaluation/questions.json", encoding="utf-8") as file:
        questions = json.load(file)

    required = {"id", "question", "category", "expected_behavior"}
    assert len(questions) == 24
    assert all(required <= set(question) for question in questions)
    assert len({question["id"] for question in questions}) == len(questions)
    assert all(question["expected_behavior"] in VALID_BEHAVIORS for question in questions)
    assert {
        question["scenario_id"]
        for question in questions
        if question.get("scenario_id")
    } == {
        "payment-timeout-v1",
        "checkout-502-v1",
        "postgres-lock-contention-v1",
        "notification-rate-limit-v1",
    }


def test_stage4_question_file_covers_filter_and_failure_boundaries():
    with open("evaluation/questions_stage4.json", encoding="utf-8") as file:
        questions = json.load(file)

    required = {"id", "question", "category", "expected_behavior"}
    assert len(questions) == 32
    assert all(required <= set(question) for question in questions)
    assert len({question["id"] for question in questions}) == len(questions)
    assert all(question["expected_behavior"] in VALID_BEHAVIORS for question in questions)
    assert {
        question["filter_case"]
        for question in questions
        if question.get("filter_case")
    } >= {
        "temporal_only",
        "service_only",
        "service_only_semantic",
        "service_and_time_semantic",
        "temporal_and_topic_semantic",
        "service_and_time_with_trace_expansion",
        "multiple_services",
        "temporal_only_empty_window",
        "unrecognized_service",
        "unsupported_time_expression",
    }


def test_behavior_classifier_output_must_be_one_known_label():
    abstain = parse_classification("abstain")
    ambiguity = parse_classification(" FLAG_AMBIGUITY ")
    invalid = parse_classification("The answer is abstain.")
    assert abstain.label == "abstain"
    assert ambiguity.label == "flag_ambiguity"
    assert invalid.label is None
    assert behavior_matches("abstain", abstain)
    assert not behavior_matches("abstain", invalid)


def test_empty_retrieval_behavior_is_deterministic():
    assert classify_empty_retrieval(None).label == "abstain"
    assert classify_empty_retrieval(
        "No chunks were found for the requested service(s): inventory-service."
    ).label == "report_no_correlated_incident"


def test_context_metrics_skip_empty_retrievals():
    assert should_score_context_metrics("reference", ["log context"])
    assert not should_score_context_metrics("reference", [])
    assert not should_score_context_metrics(None, ["log context"])


def test_evaluation_model_is_configurable_for_cerebras(monkeypatch):
    monkeypatch.setenv("EVAL_PROVIDER", "cerebras")
    monkeypatch.setenv("EVAL_MODEL", "gemma-4-31b")
    monkeypatch.setenv("EVAL_BASE_URL", "https://api.cerebras.ai/v1")
    monkeypatch.setenv("EVAL_API_KEY", "test-key")
    monkeypatch.delenv("EVAL_REASONING_EFFORT", raising=False)

    config = EvaluationLLMConfig.from_env()

    assert config.provider == "cerebras"
    assert config.model == "gemma-4-31b"
    assert config.base_url == "https://api.cerebras.ai/v1"
    assert config.request_options() == {}
