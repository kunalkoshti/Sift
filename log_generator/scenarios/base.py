"""Shared types and materialization logic for incident scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sift_common.schema import SCHEMA_VERSION, LogRecord


@dataclass(frozen=True)
class ScenarioEvent:
    """An incident event before it is assigned an absolute timestamp and host."""

    event_id: str
    service: str
    level: str
    message: str
    relative_offset_seconds: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GroundTruth:
    scenario_id: str
    root_cause_description: str
    expected_answer: str
    smoking_gun_event_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "root_cause_description": self.root_cause_description,
            "expected_answer": self.expected_answer,
            "smoking_gun_event_ids": list(self.smoking_gun_event_ids),
        }


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    trace_id: str
    events: tuple[ScenarioEvent, ...]
    ground_truth: GroundTruth

    def materialize(
        self,
        base_timestamp: datetime,
        service_catalog: dict[str, dict[str, Any]],
    ) -> list[LogRecord]:
        """Convert relative events into validated canonical log records."""

        if base_timestamp.tzinfo is None or base_timestamp.utcoffset() is None:
            raise ValueError("scenario base timestamp must be timezone-aware")

        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError(f"scenario {self.scenario_id!r} contains duplicate event IDs")
        if self.ground_truth.scenario_id != self.scenario_id:
            raise ValueError("ground truth scenario ID does not match the scenario")
        unknown_smoking_guns = set(self.ground_truth.smoking_gun_event_ids) - set(event_ids)
        if unknown_smoking_guns:
            raise ValueError(
                "ground truth references unknown events: "
                + ", ".join(sorted(unknown_smoking_guns))
            )

        offsets = [event.relative_offset_seconds for event in self.events]
        if offsets != sorted(offsets):
            raise ValueError(f"scenario {self.scenario_id!r} events must be ordered by offset")

        records: list[LogRecord] = []
        for event in self.events:
            service = service_catalog.get(event.service)
            if service is None:
                raise ValueError(f"scenario references unknown service {event.service!r}")
            host = service.get("host")
            if not isinstance(host, str) or not host:
                raise ValueError(f"service {event.service!r} has no valid host")

            metadata = {
                **event.metadata,
                "scenario_event_id": event.event_id,
                "relative_offset_seconds": event.relative_offset_seconds,
            }
            records.append(
                LogRecord(
                    schema_version=SCHEMA_VERSION,
                    timestamp=base_timestamp + timedelta(seconds=event.relative_offset_seconds),
                    level=event.level,
                    service=event.service,
                    host=host,
                    message=event.message,
                    trace_id=self.trace_id,
                    metadata=metadata,
                )
            )
        return records
