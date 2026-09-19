"""Run the reusable Stage 0 evaluation harness.

Example:
    python -m evaluation.run_eval --api-url http://localhost:8001
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import httpx
from dotenv import load_dotenv

from evaluation.behavior_classifier import (
    BehaviorClassification,
    BehaviorClassifier,
    behavior_matches,
    classify_empty_retrieval,
)
from evaluation.provider import EvaluationLLMConfig
from evaluation.ragas_metrics import score_ragas
from log_generator.scenarios import SCENARIOS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "evaluation" / "questions_stage4.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Sift's /ask endpoint")
    parser.add_argument("--api-url", default="http://localhost:8001")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--stage", default="stage_0_naive_dense")
    parser.add_argument("--run-id", type=UUID, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Seconds to wait between questions; useful for provider rate limits",
    )
    parser.add_argument(
        "--metric-delay-seconds",
        type=float,
        default=0.0,
        help="Seconds to wait before each RAGAS metric call; useful for TPM limits",
    )
    parser.add_argument("--skip-ragas", action="store_true")
    return parser.parse_args()


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("evaluation question file must contain a JSON array")
    required = {"id", "question", "category", "expected_behavior"}
    for question in questions:
        missing = required - question.keys()
        if missing:
            raise ValueError(
                f"question {question.get('id', '<unknown>')!r} is missing "
                + ", ".join(sorted(missing))
            )
    return questions


def reference_for(question: dict[str, Any]) -> str | None:
    explicit_reference = question.get("reference_answer")
    if explicit_reference:
        return explicit_reference
    scenario_id = question.get("scenario_id")
    if not scenario_id:
        return None
    try:
        return SCENARIOS[scenario_id].ground_truth.expected_answer
    except KeyError as exc:
        raise ValueError(f"question references unknown scenario {scenario_id!r}") from exc


def is_temporal_fallback_note(retrieval_note: str | None) -> bool:
    """Identify answers produced after an empty time-filter fallback."""

    normalized = (retrieval_note or "").casefold()
    return "no chunks found in the requested time range" in normalized


def fallback_was_disclosed(
    *,
    retrieval_note: str | None,
    answer: str,
) -> bool | None:
    """Check that a fallback answer discloses the requested-range miss.

    This is deliberately separate from the four behavior labels. A fallback
    answer may be a valid evidence-based answer, but it must not imply that
    the evidence came from the requested time range.
    """

    if not is_temporal_fallback_note(retrieval_note):
        return None

    normalized_answer = " ".join(answer.casefold().split())
    explicit_disclosure_terms = (
        "no logs were found",
        "no chunks were found",
        "no log entries",
        "no other events were returned",
        "no other traces were returned",
        "outside the requested time range",
        "outside the requested period",
        "not in the requested time range",
        "not from the requested time range",
        "limited to the evidence that was actually retrieved",
    )
    mentions_requested_scope = (
        "requested" in normalized_answer
        and any(
            term in normalized_answer
            for term in ("time range", "time window", "period", "window", "last week")
        )
    )
    if mentions_requested_scope and any(
        term in normalized_answer for term in explicit_disclosure_terms
    ):
        return True

    mentions_fallback_source = any(
        term in normalized_answer
        for term in (
            "full available time range",
            "full available corpus",
            "available corpus",
            "fallback",
        )
    )
    return mentions_fallback_source and mentions_requested_scope


async def ask_one(client: httpx.AsyncClient, api_url: str, question: str) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = await client.post(f"{api_url.rstrip('/')}/ask", json={"question": question})
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json(), latency_ms


async def insert_rows(
    dsn: str,
    run_id: UUID,
    run_timestamp: datetime,
    stage: str,
    rows: list[dict[str, Any]],
) -> None:
    sql = """
    INSERT INTO eval_runs (
      run_id, run_timestamp, stage_name, question_id, question_text, category,
      generated_answer, retrieved_chunk_ids, faithfulness, context_precision,
      context_recall, answer_relevancy, classified_behavior, behavior_match,
      classifier_failed, fallback_disclosed, latency_ms
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
    """
    connection = await asyncpg.connect(dsn)
    try:
        await connection.executemany(
            sql,
            [
                (
                    run_id,
                    run_timestamp,
                    stage,
                    row["question_id"],
                    row["question_text"],
                    row["category"],
                    row["generated_answer"],
                    [UUID(chunk_id) for chunk_id in row["retrieved_chunk_ids"]],
                    row["faithfulness"],
                    row["context_precision"],
                    row["context_recall"],
                    row["answer_relevancy"],
                    row["classified_behavior"],
                    row["behavior_match"],
                    row["classifier_failed"],
                    row["fallback_disclosed"],
                    row["latency_ms"],
                )
                for row in rows
            ],
        )
    finally:
        await connection.close()


async def run() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    questions = load_questions(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]

    run_id = args.run_id or uuid4()
    run_timestamp = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []

    eval_config = EvaluationLLMConfig.from_env()
    classifier_model = (
        os.getenv("EVAL_CLASSIFIER_MODEL")
        or os.getenv("BEHAVIOR_CLASSIFIER_MODEL")
        or eval_config.model
    )
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    postgres_dsn = os.environ["POSTGRES_DSN"]

    classifier = BehaviorClassifier(
        config=eval_config,
        model_name=classifier_model,
    )
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            for question_index, question in enumerate(questions):
                if question_index > 0 and args.delay_seconds > 0:
                    print(
                        f"waiting {args.delay_seconds:.1f}s before next question...",
                        flush=True,
                    )
                    await asyncio.sleep(args.delay_seconds)

                question_id = question["id"]
                question_text = question["question"]
                payload, latency_ms = await ask_one(client, args.api_url, question_text)
                answer = payload["answer"]
                retrieved_chunks = payload.get("retrieved_chunks", [])
                contexts = [chunk["content"] for chunk in retrieved_chunks]
                reference = reference_for(question)
                if retrieved_chunks:
                    try:
                        classification = await classifier.classify(
                            question_text,
                            answer,
                            payload.get("retrieval_note"),
                        )
                    except Exception as exc:
                        print(
                            f"behavior classifier failed for {question_id}: "
                            f"{type(exc).__name__}",
                            flush=True,
                        )
                        classification = BehaviorClassification(
                            raw_output=(
                                f"<classifier failure: {type(exc).__name__}>"
                            ),
                            label=None,
                        )
                else:
                    classification = classify_empty_retrieval(
                        payload.get("retrieval_note")
                    )

                if args.skip_ragas:
                    scores = {
                        "faithfulness": None,
                        "context_precision": None,
                        "context_recall": None,
                        "answer_relevancy": None,
                    }
                else:
                    scores = await score_ragas(
                        question=question_text,
                        answer=answer,
                        contexts=contexts,
                        reference=reference,
                        config=eval_config,
                        embedding_model=embedding_model,
                        metric_delay_seconds=args.metric_delay_seconds,
                        score_context_metrics=not is_temporal_fallback_note(
                            payload.get("retrieval_note")
                        ),
                    )

                row = {
                    "question_id": question_id,
                    "question_text": question_text,
                    "category": question["category"],
                    "generated_answer": answer,
                    "retrieved_chunk_ids": [chunk["id"] for chunk in retrieved_chunks],
                    **scores,
                    "classified_behavior": classification.stored_value,
                    "classifier_failed": classification.label is None,
                    "fallback_disclosed": fallback_was_disclosed(
                        retrieval_note=payload.get("retrieval_note"),
                        answer=answer,
                    ),
                    "behavior_match": (
                        None
                        if classification.label is None
                        else behavior_matches(
                            question["expected_behavior"], classification
                        )
                    ),
                    "latency_ms": latency_ms,
                }
                rows.append(row)
                classifier_status = (
                    "classifier_failure" if row["classifier_failed"] else ""
                )
                print(
                    f"{question_id}: latency={latency_ms:.1f}ms "
                    f"chunks={len(retrieved_chunks)} "
                    f"classified_behavior={row['classified_behavior']!r} "
                    f"behavior_match={row['behavior_match']} {classifier_status}"
                )
    finally:
        await classifier.close()

    await insert_rows(postgres_dsn, run_id, run_timestamp, args.stage, rows)
    print(f"stored {len(rows)} evaluation rows in eval_runs; run_id={run_id}")


if __name__ == "__main__":
    asyncio.run(run())
