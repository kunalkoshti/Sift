from datetime import datetime, timezone

from log_embedder.worker import format_content, group_records, prepare_chunk


def _record(record_id: int, timestamp: str, trace_id: str | None = None) -> dict:
    return {
        "id": record_id,
        "timestamp": datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc),
        "level": "INFO",
        "service": "payment-service",
        "message": f"message-{record_id}",
        "trace_id": trace_id,
    }


def test_groups_records_into_fixed_utc_windows_and_formats_content():
    records = [
        _record(2, "2025-12-31T23:58:30", "trace-1"),
        _record(1, "2025-12-31T23:58:00"),
    ]

    groups = group_records(records, window_seconds=60, max_records_per_chunk=25)
    chunk = prepare_chunk(groups[0])

    assert groups[0].window_start.isoformat() == "2025-12-31T23:58:00+00:00"
    assert groups[0].sub_index == 0
    assert chunk.content.splitlines() == [
        "[23:58:00] INFO payment-service: message-1",
        "[23:58:30] INFO payment-service: message-2",
    ]
    assert chunk.trace_ids == ["trace-1"]


def test_overflow_is_split_in_timestamp_and_id_order():
    records = [_record(index, "2025-12-31T23:58:00") for index in range(1, 6)]

    groups = group_records(
        records,
        window_seconds=60,
        max_records_per_chunk=2,
    )

    assert [len(group.records) for group in groups] == [2, 2, 1]
    assert [group.sub_index for group in groups] == [0, 1, 2]
    assert [list(group.records)[0]["id"] for group in groups] == [1, 3, 5]
