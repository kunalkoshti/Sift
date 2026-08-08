"""Seeded batch/live corpus generator."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from datetime import timedelta
import os
from pathlib import Path
import random
import string
import time
from typing import Any

from dotenv import load_dotenv

from log_generator.adapter import load_service_catalog
from log_generator.pipeline import merge_records, open_output, write_ground_truth, write_jsonl
from log_generator.reproducibility import GenerationContext, create_generation_context
from sift_common.schema import SCHEMA_VERSION, LogRecord
from log_generator.scenarios import SCENARIOS
from log_generator.scenarios.base import ScenarioDefinition


ROOT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ROOT_ENV_FILE)

NOISE_CATEGORY_WEIGHTS = {
    "normal": 0.80,
    "warning": 0.14,
    "error": 0.05,
    "critical": 0.01,
}
CATEGORY_LEVELS = {
    "normal": "INFO",
    "warning": "WARN",
    "error": "ERROR",
    "critical": "CRITICAL",
}

ValueProvider = Callable[[random.Random, str, str], Any]


def _identifier(prefix: str) -> ValueProvider:
    def provider(rng: random.Random, _category: str, _service_name: str) -> str:
        return f"{prefix}-{rng.randint(10000, 99999)}"

    return provider


def _duration(rng: random.Random, category: str, _service_name: str) -> int:
    ranges = {
        "normal": (20, 250),
        "warning": (500, 2500),
        "error": (2500, 8000),
        "critical": (5000, 12000),
    }
    low, high = ranges[category]
    return rng.randint(low, high)


def _status(rng: random.Random, category: str, _service_name: str) -> int:
    statuses = {
        "normal": (200, 201, 204, 302),
        "warning": (400, 404, 409, 416, 429),
        "error": (500, 502, 503, 504),
        "critical": (500, 503, 504),
    }
    return rng.choice(statuses[category])


def _percent(rng: random.Random, category: str, _service_name: str) -> int:
    ranges = {"normal": (20, 70), "warning": (75, 94), "error": (95, 100), "critical": (99, 100)}
    low, high = ranges[category]
    return rng.randint(low, high)


def _memory_mb(rng: random.Random, category: str, _service_name: str) -> int:
    ranges = {"normal": (256, 512), "warning": (600, 900), "error": (900, 1024), "critical": (1024, 1200)}
    low, high = ranges[category]
    return rng.randint(low, high)


def _value_providers() -> dict[str, ValueProvider]:
    """Return deterministic providers for every configured template field."""

    return {
        "cart_id": _identifier("cart"),
        "channel": lambda rng, _category, _service: rng.choice(("email", "sms", "push")),
        "checkout_id": _identifier("checkout"),
        "client_id": _identifier("client"),
        "client_service": lambda rng, _category, _service: rng.choice(
            ("payment-service", "checkout-service", "inventory-service")
        ),
        "customer_id": _identifier("customer"),
        "duration_ms": _duration,
        "item_count": lambda rng, _category, _service: rng.randint(1, 8),
        "lag_ms": lambda rng, _category, _service: rng.randint(2, 800),
        "memory_mb": _memory_mb,
        "message_id": _identifier("message"),
        "method": lambda rng, _category, _service: rng.choice(("GET", "POST", "PUT", "DELETE")),
        "order_id": _identifier("order"),
        "path": lambda rng, _category, _service: rng.choice(
            ("/checkout", "/payments", "/orders", "/inventory", "/health")
        ),
        "payment_id": _identifier("payment"),
        "percent": _percent,
        "query_name": lambda rng, _category, _service: rng.choice(
            ("order_lookup", "payment_insert", "inventory_reservation", "customer_lookup")
        ),
        "queue_depth": lambda rng, category, _service: rng.randint(
            0 if category == "normal" else 100, 500 if category == "normal" else 10000
        ),
        "restart_count": lambda rng, _category, _service: rng.randint(1, 4),
        "status": _status,
        "uptime_minutes": lambda rng, _category, _service: rng.randint(5, 240),
        "warehouse": lambda rng, _category, _service: rng.choice(("us-east-1", "us-west-2", "eu-west-1")),
    }


VALUE_PROVIDERS = _value_providers()


def _placeholder_names(template: str) -> list[str]:
    names: list[str] = []
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name is not None and field_name not in names:
            names.append(field_name)
    return names


def _choose_category(rng: random.Random, service: dict[str, Any]) -> str:
    available = [
        category
        for category in NOISE_CATEGORY_WEIGHTS
        if service.get(f"{category}_messages")
    ]
    if not available:
        raise ValueError(f"service {service.get('name')!r} has no message templates")
    weights = [NOISE_CATEGORY_WEIGHTS[category] for category in available]
    return rng.choices(available, weights=weights, k=1)[0]


def _render_template(
    template: str,
    *,
    rng: random.Random,
    category: str,
    service_name: str,
) -> tuple[str, dict[str, Any]]:
    values: dict[str, Any] = {}
    for name in _placeholder_names(template):
        provider = VALUE_PROVIDERS.get(name)
        if provider is None:
            raise ValueError(f"no value provider registered for template placeholder {{{name}}}")
        values[name] = provider(rng, category, service_name)
    return template.format(**values), values


def _generate_noise(
    context: GenerationContext,
    service_catalog: dict[str, dict[str, Any]],
    count: int,
) -> list[LogRecord]:
    if count < 2:
        raise ValueError("noise count must be at least 2")

    services = tuple(service_catalog.values())
    records: list[LogRecord] = []
    noise_start_seconds = int(os.environ["GENERATOR_NOISE_START_SECONDS"])
    noise_end_seconds = int(os.environ["GENERATOR_NOISE_END_SECONDS"])
    if noise_end_seconds <= noise_start_seconds:
        raise ValueError("GENERATOR_NOISE_END_SECONDS must be greater than the start value")
    total_window = noise_end_seconds - noise_start_seconds
    for index in range(count):
        service = context.rng.choice(services)
        category = _choose_category(context.rng, service)
        templates = service[f"{category}_messages"]
        template = context.rng.choice(templates)
        message, values = _render_template(
            template,
            rng=context.rng,
            category=category,
            service_name=service["name"],
        )
        position = index / (count - 1)
        jitter = context.rng.uniform(-2.0, 2.0)
        offset = noise_start_seconds + total_window * position + jitter
        records.append(
            LogRecord(
                schema_version=SCHEMA_VERSION,
                timestamp=context.base_timestamp + timedelta(seconds=offset),
                level=CATEGORY_LEVELS[category],
                service=service["name"],
                host=service["host"],
                message=message,
                trace_id=None,
                metadata={
                    "source": "seeded-template-noise",
                    "template_category": category,
                    "template_values": values,
                },
            )
        )
    records.sort(key=lambda record: record.timestamp)
    return records


def generate_event_stream(
    *,
    context: GenerationContext,
    service_catalog: dict[str, dict[str, Any]],
    scenario: ScenarioDefinition,
    noise_count: int,
) -> Iterator[LogRecord]:
    """Build the one event iterator consumed by both output modes."""

    noise = _generate_noise(context, service_catalog, noise_count)
    scenario_records = scenario.materialize(context.base_timestamp, service_catalog)
    yield from merge_records(noise, scenario_records)


def _write_live(records: Iterator[LogRecord], destination: Any, delay_scale: float) -> None:
    previous_timestamp = None
    for record in records:
        if previous_timestamp is not None:
            delay = (record.timestamp - previous_timestamp).total_seconds() * delay_scale
            if delay > 0:
                time.sleep(delay)
        destination.write(record.model_dump_json() + "\n")
        destination.flush()
        previous_timestamp = record.timestamp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("batch", "live"), default="batch")
    parser.add_argument("--seed", type=int, help="seed for reproducible noise and timing")
    parser.add_argument("--noise-count", type=int, default=60)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="payment-timeout-v1")
    parser.add_argument("--output", default="-", help="JSONL output path, or - for stdout")
    parser.add_argument("--ground-truth", required=True, help="separate ground-truth JSON path")
    parser.add_argument(
        "--delay-scale",
        type=float,
        default=1.0,
        help="live-mode timestamp delay multiplier; use 0 for immediate live output",
    )
    args = parser.parse_args()

    if args.delay_scale < 0:
        parser.error("--delay-scale must be non-negative")
    if args.ground_truth == "-":
        parser.error("--ground-truth must be a separate file path, not stdout")
    if args.output != "-" and args.output == args.ground_truth:
        parser.error("--output and --ground-truth must be different paths")

    service_catalog = load_service_catalog()
    scenario = SCENARIOS[args.scenario]
    context = create_generation_context(args.seed)

    try:
        with open_output(args.ground_truth) as ground_truth_stream:
            write_ground_truth(scenario.ground_truth.as_dict(), ground_truth_stream)
        records = generate_event_stream(
            context=context,
            service_catalog=service_catalog,
            scenario=scenario,
            noise_count=args.noise_count,
        )
        with open_output(args.output) as output_stream:
            if args.mode == "batch":
                write_jsonl(records, output_stream)
            else:
                _write_live(records, output_stream, args.delay_scale)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
