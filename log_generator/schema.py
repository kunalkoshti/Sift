from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = "1.0"
LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]


class LogRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: str
    timestamp: datetime
    level: LogLevel
    service: str = Field(min_length=1)
    host: str = Field(min_length=1)
    message: str = Field(min_length=1)
    trace_id: str | None
    metadata: dict[str, Any]

    @field_validator("schema_version")
    @classmethod
    def schema_version_must_match(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        return value

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

    @field_validator("trace_id")
    @classmethod
    def trace_id_must_not_be_empty(cls, value: str | None) -> str | None:
        if value == "":
            raise ValueError("trace_id must be null or a non-empty string")
        return value
