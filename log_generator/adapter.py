
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from sift_common.schema import SCHEMA_VERSION, LogRecord


DEFAULT_SERVICES_PATH = Path(__file__).with_name("services.json")
FLOG_TIMESTAMP_FORMAT = "%d/%b/%Y:%H:%M:%S %z"
REQUIRED_FLOG_FIELDS = {
    "host",
    "user-identifier",
    "datetime",
    "method",
    "request",
    "protocol",
    "status",
    "bytes",
    "referer",
}
SUPPORTED_GENERATORS = {"flog", "template"}


class AdapterError(ValueError):
    """Raised when a raw flog record cannot be adapted."""


def load_service_catalog(path: Path = DEFAULT_SERVICES_PATH) -> dict[str, dict[str, Any]]:
    """Load and minimally validate the configured service names and hosts."""

    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"could not load service configuration {path}: {exc}") from exc

    services = document.get("services") if isinstance(document, dict) else None
    if not isinstance(services, list) or not services:
        raise AdapterError("service configuration must contain a non-empty services list")

    catalog: dict[str, dict[str, Any]] = {}
    for service in services:
        if not isinstance(service, dict):
            raise AdapterError("every service configuration entry must be an object")

        name = service.get("name")
        host = service.get("host")
        generator = service.get("generator")
        if not isinstance(name, str) or not name.strip():
            raise AdapterError("every service configuration entry needs a non-empty name")
        if not isinstance(host, str) or not host.strip():
            raise AdapterError(f"service {name!r} needs a non-empty host")
        if generator not in SUPPORTED_GENERATORS:
            expected = ", ".join(sorted(SUPPORTED_GENERATORS))
            raise AdapterError(
                f"service {name!r} needs generator={expected!r}; got {generator!r}"
            )
        if name in catalog:
            raise AdapterError(f"duplicate service name: {name}")

        catalog[name] = service

    return catalog


def _level_from_status(status: int) -> str:
    if status >= 500:
        return "ERROR"
    if status >= 400:
        return "WARN"
    return "INFO"


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise AdapterError("flog datetime must be a string")
    try:
        return datetime.strptime(value, FLOG_TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise AdapterError(
            f"unsupported flog datetime {value!r}; expected {FLOG_TIMESTAMP_FORMAT!r}"
        ) from exc


def adapt_flog_record(
    raw_record: Mapping[str, Any],
    service: Mapping[str, Any],
) -> LogRecord:
    """Transform one parsed flog object into a validated ``LogRecord``."""

    missing = REQUIRED_FLOG_FIELDS - raw_record.keys()
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise AdapterError(f"flog record is missing required fields: {missing_fields}")

    service_name = service.get("name")
    service_host = service.get("host")
    if not isinstance(service_name, str) or not service_name.strip():
        raise AdapterError("service configuration has an invalid name")
    if not isinstance(service_host, str) or not service_host.strip():
        raise AdapterError(f"service {service_name!r} has an invalid host")

    status = raw_record["status"]
    if isinstance(status, bool) or not isinstance(status, int):
        raise AdapterError("flog status must be an integer")

    method = raw_record["method"]
    request = raw_record["request"]
    if not isinstance(method, str) or not isinstance(request, str):
        raise AdapterError("flog method and request must be strings")

    metadata: dict[str, Any] = {
        "source": "flog",
        "source_host": raw_record["host"],
        "user_identifier": raw_record["user-identifier"],
        "method": method,
        "request": request,
        "protocol": raw_record["protocol"],
        "status": status,
        "bytes": raw_record["bytes"],
        "referer": raw_record["referer"],
    }

    extra_fields = set(raw_record) - REQUIRED_FLOG_FIELDS
    if extra_fields:
        metadata["raw_extra"] = {key: raw_record[key] for key in sorted(extra_fields)}

    return LogRecord(
        schema_version=SCHEMA_VERSION,
        timestamp=_parse_timestamp(raw_record["datetime"]),
        level=_level_from_status(status),
        service=service_name,
        host=service_host,
        message=f"{method} {request} returned HTTP {status}",
        trace_id=None,
        metadata=metadata,
    )


def adapt_flog_line(line: str, service: Mapping[str, Any]) -> LogRecord:
    """Parse and adapt one JSONL line."""

    try:
        raw_record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise AdapterError(f"invalid JSON: {exc.msg}") from exc

    if not isinstance(raw_record, dict):
        raise AdapterError("each flog JSONL line must contain an object")

    return adapt_flog_record(raw_record, service)


def adapt_flog_stream(
    lines: Iterable[str],
    service: Mapping[str, Any],
) -> Iterable[LogRecord]:
    """Adapt every non-empty line, preserving input order."""

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            yield adapt_flog_line(line, service)
        except AdapterError as exc:
            raise AdapterError(f"line {line_number}: {exc}") from exc


def _open_text(path: str, mode: str) -> nullcontext[TextIO] | Any:
    if path == "-":
        import sys

        stream = sys.stdin if "r" in mode else sys.stdout
        return nullcontext(stream)
    return Path(path).open(mode, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="-", help="flog JSONL input path, or - for stdin")
    parser.add_argument("--output", default="-", help="adapted JSONL output path, or - for stdout")
    parser.add_argument("--service", required=True, help="configured service name to assign")
    parser.add_argument(
        "--services",
        default=str(DEFAULT_SERVICES_PATH),
        help="service configuration JSON path",
    )
    args = parser.parse_args()

    catalog = load_service_catalog(Path(args.services))
    if args.service not in catalog:
        available = ", ".join(sorted(catalog))
        parser.error(f"unknown service {args.service!r}; choose from: {available}")
    if catalog[args.service]["generator"] != "flog":
        generator = catalog[args.service]["generator"]
        parser.error(
            f"service {args.service!r} is not flog-compatible "
            f"(generator={generator!r}); use the template generator instead"
        )

    try:
        with _open_text(args.input, "r") as input_stream, _open_text(args.output, "w") as output_stream:
            for record in adapt_flog_stream(input_stream, catalog[args.service]):
                output_stream.write(record.model_dump_json() + "\n")
    except (OSError, AdapterError) as exc:
        parser.error(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
