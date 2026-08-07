import pytest

from log_consumer.consumer import ConsumerConfig, load_table_schema, parse_record
from sift_common.schema import LogRecord


def _record_json() -> str:
    return LogRecord(
        schema_version="1.0",
        timestamp="2025-12-31T23:57:00Z",
        level="ERROR",
        service="payment-service",
        host="payment-service-1",
        message="Payment authorization timed out",
        trace_id="trace-payment-timeout-001",
        metadata={"scenario_event_id": "payment-timeout-04"},
    ).model_dump_json()


def test_parse_record_revalidates_canonical_log_record():
    record = parse_record({"record": _record_json()})

    assert record.service == "payment-service"
    assert record.trace_id == "trace-payment-timeout-001"


def test_parse_record_rejects_missing_record_field():
    with pytest.raises(ValueError, match="missing the record field"):
        parse_record({})


def test_schema_contains_required_columns_and_unique_guard():
    schema = load_table_schema()

    for column in (
        "schema_version",
        "timestamp",
        "level",
        "service",
        "host",
        "message",
        "trace_id",
        "metadata",
        "redis_entry_id",
        "received_at",
    ):
        assert column in schema
    assert "redis_entry_id TEXT NOT NULL UNIQUE" in schema


def test_consumer_config_reads_environment(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://example:6379/0")
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://example/sift")
    monkeypatch.setenv("REDIS_STREAM", "test-stream")
    monkeypatch.setenv("REDIS_GROUP", "test-group")
    monkeypatch.setenv("REDIS_CONSUMER", "test-consumer")

    config = ConsumerConfig.from_env()

    assert config.redis_url == "redis://example:6379/0"
    assert config.postgres_dsn == "postgresql://example/sift"
    assert config.stream == "test-stream"
    assert config.group == "test-group"
    assert config.consumer == "test-consumer"
