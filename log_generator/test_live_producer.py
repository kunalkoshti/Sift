from datetime import datetime, timezone

from log_generator.adapter import load_service_catalog
from log_generator.generator import generate_live_record
from log_generator.live_producer import chunked
from log_generator.reproducibility import create_generation_context


def test_generate_live_record_is_schema_valid_and_marked_as_live_noise():
    context = create_generation_context(seed=123)
    record = generate_live_record(
        context,
        load_service_catalog(),
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert record.timestamp.isoformat() == "2026-01-01T00:00:00+00:00"
    assert record.trace_id is None
    assert record.metadata["source"] == "live-template-noise"


def test_chunked_preserves_records_and_batch_boundaries():
    context = create_generation_context(seed=123)
    records = [
        generate_live_record(context, load_service_catalog())
        for _ in range(5)
    ]

    batches = list(chunked(records, 2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [record for batch in batches for record in batch] == records
