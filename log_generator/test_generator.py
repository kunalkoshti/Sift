import io
import json

import pytest

from log_generator.adapter import load_service_catalog
from log_generator.generator import _write_live, generate_event_stream
from log_generator.pipeline import write_ground_truth, write_jsonl
from log_generator.reproducibility import create_generation_context
from sift_common.schema import LogRecord
from log_generator.scenarios import SCENARIOS


def _records(seed: int, scenario_id: str = "payment-timeout-v1") -> list[LogRecord]:
    catalog = load_service_catalog()
    scenario = SCENARIOS[scenario_id]
    return list(
        generate_event_stream(
            context=create_generation_context(seed),
            service_catalog=catalog,
            scenario=scenario,
            noise_count=60,
        )
    )


def _jsonl(records: list[LogRecord]) -> str:
    output = io.StringIO()
    write_jsonl(records, output)
    return output.getvalue()


@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
def test_records_validate_and_scenario_is_interleaved(scenario_id: str) -> None:
    records = _records(123, scenario_id)
    required_fields = set(LogRecord.model_fields)
    serialized = [json.loads(record.model_dump_json()) for record in records]

    assert all(set(record) == required_fields for record in serialized)
    assert all(LogRecord.model_validate(record) for record in serialized)
    assert all(left.timestamp <= right.timestamp for left, right in zip(records, records[1:]))

    scenario = SCENARIOS[scenario_id]
    scenario_records = [record for record in records if record.trace_id == scenario.trace_id]
    expected_ids = [event.event_id for event in scenario.events]
    actual_ids = [record.metadata["scenario_event_id"] for record in scenario_records]
    positions = [index for index, record in enumerate(records) if record.trace_id == scenario.trace_id]

    assert actual_ids == expected_ids
    assert positions[0] > 0
    assert positions[-1] < len(records) - 1

    logs = _jsonl(records)
    ground_truth = io.StringIO()
    write_ground_truth(scenario.ground_truth.as_dict(), ground_truth)
    assert "root_cause_description" not in logs
    assert json.loads(ground_truth.getvalue())["scenario_id"] == scenario_id
    assert set(scenario.ground_truth.smoking_gun_event_ids).issubset(set(actual_ids))


def test_same_seed_is_identical_and_live_uses_same_records() -> None:
    batch_records = _records(123)
    same_seed_records = _records(123)
    different_seed_records = _records(124)

    assert _jsonl(batch_records) == _jsonl(same_seed_records)
    assert _jsonl(batch_records) != _jsonl(different_seed_records)

    live_output = io.StringIO()
    _write_live(iter(batch_records), live_output, delay_scale=0)
    assert live_output.getvalue() == _jsonl(batch_records)


def test_seeded_timestamp_is_fixed() -> None:
    first = _records(123)[0].timestamp
    second = _records(456)[0].timestamp
    assert create_generation_context(123).base_timestamp == create_generation_context(456).base_timestamp
    assert first.tzinfo is not None
    assert second.tzinfo is not None
