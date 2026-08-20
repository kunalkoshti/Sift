import json

from evaluation.behavior_classifier import (
    VALID_BEHAVIORS,
    behavior_matches,
    parse_classification,
)
from evaluation.provider import EvaluationLLMConfig


def test_question_file_is_structured_and_covers_all_scenarios():
    with open("evaluation/questions.json", encoding="utf-8") as file:
        questions = json.load(file)

    required = {"id", "question", "category", "expected_behavior"}
    assert len(questions) == 20
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


def test_behavior_classifier_output_must_be_one_known_label():
    abstain = parse_classification("abstain")
    ambiguity = parse_classification(" FLAG_AMBIGUITY ")
    invalid = parse_classification("The answer is abstain.")
    assert abstain.label == "abstain"
    assert ambiguity.label == "flag_ambiguity"
    assert invalid.label is None
    assert behavior_matches("abstain", abstain)
    assert not behavior_matches("abstain", invalid)


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
