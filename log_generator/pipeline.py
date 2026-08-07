"""Shared merge and output functions for generated log records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator, TextIO

from schema import LogRecord


def merge_records(
    noise_records: Iterable[LogRecord],
    scenario_records: Iterable[LogRecord],
) -> list[LogRecord]:
    """Interleave scenario records into the noise timestamp window.

    Input order becomes the stable sequence number used to break equal-time
    ties. Noise records are supplied first, followed by scenario records.
    """

    noise = list(noise_records)
    scenario = list(scenario_records)
    if not noise:
        raise ValueError("cannot merge a scenario into an empty noise stream")

    first_noise_timestamp = min(record.timestamp for record in noise)
    last_noise_timestamp = max(record.timestamp for record in noise)
    outside_window = [
        record
        for record in scenario
        if record.timestamp < first_noise_timestamp or record.timestamp > last_noise_timestamp
    ]
    if outside_window:
        raise ValueError("scenario events must fall within the noise timestamp window")

    indexed_records = [
        (record.timestamp, sequence, record)
        for sequence, record in enumerate((*noise, *scenario))
    ]
    indexed_records.sort(key=lambda item: (item[0], item[1]))
    return [record for _, _, record in indexed_records]


def write_jsonl(records: Iterable[LogRecord], destination: TextIO) -> None:
    """Write only validated log records to JSONL."""

    for record in records:
        destination.write(record.model_dump_json() + "\n")


def write_ground_truth(ground_truth: Mapping[str, Any], destination: TextIO) -> None:
    """Write scenario ground truth as a separate JSON document."""

    json.dump(ground_truth, destination, indent=2, sort_keys=True)
    destination.write("\n")


@contextmanager
def open_output(path: str) -> Iterator[TextIO]:
    """Yield stdout for ``-`` or a UTF-8 file handle for a path."""

    if path == "-":
        yield sys.stdout
        return

    with Path(path).open("w", encoding="utf-8") as stream:
        yield stream
