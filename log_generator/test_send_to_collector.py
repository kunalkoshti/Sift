from __future__ import annotations

from log_generator.send_to_collector import chunked, send_batches
from sift_common.schema import LogRecord


def _record(index: int) -> LogRecord:
    return LogRecord(
        schema_version="1.0",
        timestamp="2026-01-01T00:00:00Z",
        level="INFO",
        service="api-gateway",
        host="api-gateway-1",
        message=f"message {index}",
        trace_id=None,
        metadata={"index": index},
    )


class FakeResponse:
    status_code = 202
    text = ""
    is_error = False

    def __init__(self, accepted: int) -> None:
        self.accepted = accepted

    def json(self) -> dict[str, int]:
        return {"accepted": self.accepted}


class FakeClient:
    requests: list[tuple[str, list[dict[str, object]]]] = []

    def __init__(self, **_: object) -> None:
        self.requests = []
        FakeClient.requests = self.requests

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def post(self, path: str, json: list[dict[str, object]]) -> FakeResponse:
        self.requests.append((path, json))
        return FakeResponse(len(json))


def test_chunked_preserves_order() -> None:
    records = [_record(index) for index in range(5)]

    batches = list(chunked(records, 2))

    assert [[record.metadata["index"] for record in batch] for batch in batches] == [
        [0, 1],
        [2, 3],
        [4],
    ]


def test_send_batches_posts_json_lists_and_counts_acceptance(monkeypatch) -> None:
    monkeypatch.setattr("log_generator.send_to_collector.httpx.Client", FakeClient)
    records = [_record(index) for index in range(5)]

    accepted = send_batches(
        records,
        collector_url="http://collector",
        batch_size=2,
        timeout=1,
    )

    assert accepted == 5
    assert [len(payload) for path, payload in FakeClient.requests] == [2, 2, 1]
    assert all(path == "/ingest" for path, _ in FakeClient.requests)
    assert FakeClient.requests[0][1][0]["message"] == "message 0"
    assert FakeClient.requests[0][1][1]["metadata"] == {"index": 1}
