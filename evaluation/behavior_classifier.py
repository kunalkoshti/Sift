"""Independent LLM classifier for expected QA response behavior."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

from evaluation.provider import EvaluationLLMConfig


VALID_BEHAVIORS = frozenset(
    {
        "answer_with_evidence",
        "abstain",
        "flag_ambiguity",
        "report_no_correlated_incident",
    }
)

CLASSIFIER_SYSTEM_PROMPT = """Classify a log-QA system response into exactly one label.

Labels:
- answer_with_evidence: directly answers the question from available log evidence.
- abstain: declines because the logs cannot support the requested answer or the
  question is outside the corpus.
- flag_ambiguity: explicitly states that multiple incidents or causes are
  plausible and the logs do not uniquely identify one.
- report_no_correlated_incident: states that no notable or trace-correlated
  incident was found for the requested service, while distinguishing that from a
  generic lack of information.

Use report_no_correlated_incident only when the answer affirmatively reports
that no notable or trace-correlated incident was found for the named service.
Use abstain when the answer says the logs lack enough information, including
for unsupported aggregate questions.

Return exactly one label, lowercase, with no explanation or punctuation.
"""


@dataclass(frozen=True)
class BehaviorClassification:
    """Raw model response and its parsed label, if it obeyed the contract."""

    raw_output: str
    label: str | None

    @property
    def stored_value(self) -> str:
        """Keep unexpected raw output visible in eval_runs for debugging."""

        return self.raw_output or "<empty classifier output>"


def behavior_matches(expected_behavior: str, classification: BehaviorClassification) -> bool:
    """Apply the safe policy: invalid classifier output is always a mismatch."""

    return classification.label is not None and classification.label == expected_behavior


def parse_classification(raw_output: str) -> BehaviorClassification:
    cleaned = raw_output.strip()
    normalized = cleaned.casefold()
    label = normalized if normalized in VALID_BEHAVIORS else None
    return BehaviorClassification(raw_output=cleaned, label=label)


async def classify_behavior(
    *,
    question: str,
    answer: str,
    config: EvaluationLLMConfig,
    model_name: str | None = None,
) -> BehaviorClassification:
    """Make one forced single-label classification call, separate from RAGAS."""

    classifier = BehaviorClassifier(
        config=config,
        model_name=model_name,
    )
    try:
        return await classifier.classify(question, answer)
    finally:
        await classifier.close()


class BehaviorClassifier:
    """Reusable classifier client for one evaluation harness process."""

    def __init__(
        self,
        *,
        config: EvaluationLLMConfig,
        model_name: str | None = None,
    ):
        self.config = config
        self.model_name = model_name or config.model
        self.client = config.build_client()

    async def classify(self, question: str, answer: str) -> BehaviorClassification:
        """Classify one question/answer pair with a deterministic single-label prompt."""

        completion = await self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            max_tokens=64,
            **self.config.request_options(),
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nSystem answer:\n{answer}",
                },
            ],
        )
        raw_output = completion.choices[0].message.content or ""
        return parse_classification(raw_output)

    async def close(self) -> None:
        await self.client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually test the behavior classifier")
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer", required=True)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    result = await classify_behavior(
        question=args.question,
        answer=args.answer,
        config=EvaluationLLMConfig.from_env(),
        model_name=(
            os.getenv("EVAL_CLASSIFIER_MODEL")
            or os.getenv("BEHAVIOR_CLASSIFIER_MODEL")
        ),
    )
    print(f"raw_output={result.stored_value!r}")
    print(f"parsed_label={result.label!r}")


if __name__ == "__main__":
    asyncio.run(main())
