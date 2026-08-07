"""Generate a batch corpus and send it to log_collector in small batches."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import httpx

from log_generator.adapter import load_service_catalog
from log_generator.generator import generate_event_stream
from log_generator.pipeline import write_ground_truth, write_jsonl
from log_generator.reproducibility import create_generation_context
from log_generator.scenarios import SCENARIOS
from sift_common.schema import LogRecord


def chunked(records: list[LogRecord], batch_size: int) -> Iterator[list[LogRecord]]:
    """Yield records in bounded batches while preserving their order."""

    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def send_batches(
    records: list[LogRecord],
    *,
    collector_url: str,
    batch_size: int,
    timeout: float,
) -> int:
    """POST all records to the collector and return the accepted count."""

    accepted_total = 0
    with httpx.Client(base_url=collector_url.rstrip("/"), timeout=timeout) as client:
        for batch_number, batch in enumerate(chunked(records, batch_size), start=1):
            payload = [json.loads(record.model_dump_json()) for record in batch]
            response = client.post("/ingest", json=payload)
            if response.is_error:
                raise RuntimeError(
                    f"collector rejected batch {batch_number}: "
                    f"HTTP {response.status_code} {response.text}"
                )

            result = response.json()
            accepted = result.get("accepted")
            if accepted != len(batch):
                raise RuntimeError(
                    f"collector accepted {accepted} records for batch {batch_number}; "
                    f"expected {len(batch)}"
                )
            accepted_total += accepted
            print(f"sent batch {batch_number}: {accepted} records")

    return accepted_total


def generate_batch(seed: int | None, scenario_id: str, noise_count: int) -> list[LogRecord]:
    scenario = SCENARIOS[scenario_id]
    return list(
        generate_event_stream(
            context=create_generation_context(seed),
            service_catalog=load_service_catalog(),
            scenario=scenario,
            noise_count=noise_count,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-url", default="http://127.0.0.1:8000")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="payment-timeout-v1")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--noise-count", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True, help="local JSONL batch output")
    parser.add_argument("--ground-truth", type=Path, help="optional separate ground-truth output")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    try:
        records = generate_batch(args.seed, args.scenario, args.noise_count)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output:
            write_jsonl(records, output)

        if args.ground_truth:
            args.ground_truth.parent.mkdir(parents=True, exist_ok=True)
            with args.ground_truth.open("w", encoding="utf-8") as truth_output:
                write_ground_truth(SCENARIOS[args.scenario].ground_truth.as_dict(), truth_output)

        accepted = send_batches(
            records,
            collector_url=args.collector_url,
            batch_size=args.batch_size,
            timeout=args.timeout,
        )
    except (OSError, RuntimeError, ValueError, httpx.HTTPError) as exc:
        parser.error(str(exc))

    print(f"generated {len(records)} records in {args.output}")
    print(f"collector accepted {accepted} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
