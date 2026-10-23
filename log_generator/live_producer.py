"""Continuously generate template logs and send them to log-collector."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import httpx

from log_generator.adapter import load_service_catalog
from log_generator.generator import generate_event_stream, generate_live_record
from log_generator.pipeline import write_ground_truth
from log_generator.reproducibility import create_generation_context
from log_generator.scenarios import SCENARIOS
from sift_common.schema import LogRecord


def chunked(records: list[LogRecord], batch_size: int):
    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def post_batch(
    client: httpx.Client,
    records: list[LogRecord],
    *,
    retry_count: int,
    retry_delay_seconds: float,
) -> int:
    payload = [json.loads(record.model_dump_json()) for record in records]
    last_error: Exception | None = None
    for attempt in range(retry_count + 1):
        try:
            response = client.post("/ingest", json=payload)
            response.raise_for_status()
            accepted = response.json().get("accepted")
            if accepted != len(records):
                raise RuntimeError(
                    f"collector accepted {accepted} records; expected {len(records)}"
                )
            return int(accepted)
        except (httpx.HTTPError, RuntimeError) as exc:
            last_error = exc
            if attempt < retry_count:
                time.sleep(retry_delay_seconds)

    raise RuntimeError(f"could not send live batch: {last_error}") from last_error


def send_records(
    client: httpx.Client,
    records: list[LogRecord],
    *,
    batch_size: int,
    retry_count: int,
    retry_delay_seconds: float,
) -> int:
    accepted = 0
    for batch in chunked(records, batch_size):
        accepted += post_batch(
            client,
            batch,
            retry_count=retry_count,
            retry_delay_seconds=retry_delay_seconds,
        )
    return accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-url", default="http://127.0.0.1:8000")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--initial-scenario",
        choices=sorted(SCENARIOS),
        help="send one scenario corpus before continuous noise begins",
    )
    parser.add_argument("--initial-noise-count", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="number of live batches to send; 0 runs until interrupted",
    )
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    args = parser.parse_args()

    if args.initial_noise_count < 2:
        parser.error("--initial-noise-count must be at least 2")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if args.max_batches < 0:
        parser.error("--max-batches must be non-negative")
    if args.timeout <= 0 or args.retry_count < 0 or args.retry_delay_seconds < 0:
        parser.error("timeout, retry count, and retry delay must be valid")

    service_catalog = load_service_catalog()
    context = create_generation_context(args.seed)
    client = httpx.Client(base_url=args.collector_url.rstrip("/"), timeout=args.timeout)
    total_accepted = 0
    live_batch_number = 0
    last_timestamp: datetime | None = None

    try:
        if args.initial_scenario:
            scenario = SCENARIOS[args.initial_scenario]
            initial_records = list(
                generate_event_stream(
                    context=context,
                    service_catalog=service_catalog,
                    scenario=scenario,
                    noise_count=args.initial_noise_count,
                )
            )
            total_accepted += send_records(
                client,
                initial_records,
                batch_size=args.batch_size,
                retry_count=args.retry_count,
                retry_delay_seconds=args.retry_delay_seconds,
            )
            if args.ground_truth:
                args.ground_truth.parent.mkdir(parents=True, exist_ok=True)
                with args.ground_truth.open("w", encoding="utf-8") as output:
                    write_ground_truth(scenario.ground_truth.as_dict(), output)
            print(f"sent initial scenario: {len(initial_records)} records")

        while args.max_batches == 0 or live_batch_number < args.max_batches:
            records: list[LogRecord] = []
            for _ in range(args.batch_size):
                timestamp = datetime.now(timezone.utc)
                if last_timestamp is not None and timestamp <= last_timestamp:
                    timestamp = last_timestamp + timedelta(microseconds=1)
                record = generate_live_record(
                    context,
                    service_catalog,
                    timestamp=timestamp,
                )
                records.append(record)
                last_timestamp = record.timestamp

            accepted = send_records(
                client,
                records,
                batch_size=args.batch_size,
                retry_count=args.retry_count,
                retry_delay_seconds=args.retry_delay_seconds,
            )
            live_batch_number += 1
            total_accepted += accepted
            print(
                f"sent live batch {live_batch_number}: {accepted} records; "
                f"total={total_accepted}",
                flush=True,
            )
            if args.max_batches == 0 or live_batch_number < args.max_batches:
                time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("live producer stopped")
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
