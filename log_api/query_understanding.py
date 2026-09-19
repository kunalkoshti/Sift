"""Deterministic query parsing for service and time filters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
import logging
import re
from typing import Sequence


logger = logging.getLogger(__name__)

DEFAULT_SERVICE_CATALOG = (
    "api-gateway",
    "checkout-service",
    "payment-service",
    "inventory-service",
    "notification-service",
    "nginx",
    "postgres",
)

_MIDNIGHT_RE = re.compile(r"\b(?:around|at)\s+midnight\b", re.IGNORECASE)
_CLOCK_RE = re.compile(
    r"\b(?:around|at)\s+(?P<hour>\d{1,2})"
    r"(?:(?::(?P<minute>\d{2}))\s*(?P<ampm>a\.m\.|p\.m\.|am|pm)"
    r"|\s*(?P<oclock>o['’]?clock)\b|\s*(?P<bare_ampm>a\.m\.|p\.m\.|am|pm))",
    re.IGNORECASE,
)
_TEMPORAL_WORD_RE = re.compile(
    r"\b(?:last\s+night|last\s+week|tonight|this\s+morning|yesterday|today|"
    r"around\s+midnight|at\s+midnight|around\s+\d{1,2}(?::\d{2})?\s*(?:a\.m\.|p\.m\.|am|pm)|"
    r"at\s+\d{1,2}(?::\d{2})?\s*(?:a\.m\.|p\.m\.|am|pm)|"
    r"around\s+\d{1,2}\s+o['’]?clock|at\s+\d{1,2}\s+o['’]?clock)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_TIME_RE = re.compile(
    r"\b(?:"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:minute|minutes|hour|hours|day|days|week|weeks)\s+ago|"
    r"last\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"month|year)|"
    r"this\s+(?:week|month|year)|tomorrow|"
    r"the\s+day\s+before\s+yesterday"
    r")\b",
    re.IGNORECASE,
)
_UNKNOWN_SERVICE_PATTERNS = (
    re.compile(
        r"\bdid\s+(?P<name>[a-z][a-z0-9_-]{1,})\s+have\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:what\s+happened|logs?|incidents?|errors?|issues?)\s+"
        r"(?:in|from|for|about)\s+(?:the\s+)?"
        r"(?P<name>[a-z][a-z0-9_-]{1,})\b",
        re.IGNORECASE,
    ),
)
_NON_SERVICE_WORDS = {
    "a",
    "an",
    "any",
    "company",
    "logs",
    "service",
    "the",
}


@dataclass(frozen=True)
class ParsedQuery:
    original: str
    intent: str
    time_range: tuple[datetime, datetime] | None
    service_filter: list[str]
    unsupported_service: str | None = None
    unsupported_time_expression: str | None = None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("query time anchor must be timezone-aware")
    return value.astimezone(timezone.utc)


def _day_start(value: datetime) -> datetime:
    value = _utc(value)
    return datetime.combine(value.date(), time.min, tzinfo=timezone.utc)


def resolve_service_filter(
    question: str,
    service_catalog: Sequence[str] = DEFAULT_SERVICE_CATALOG,
) -> list[str]:
    """Return explicitly service-scoped canonical names in query order.

    Full application service names such as ``payment-service`` and
    ``payment service`` are explicit filters. Short application words such as
    ``payment`` and ``checkout`` are treated as topic vocabulary unless the
    question includes the word ``service``. This prevents questions about
    payment or checkout incidents from being narrowed accidentally. Standalone
    infrastructure identifiers such as ``postgres`` and ``nginx`` remain
    service filters because they are unambiguous in this catalog.
    """

    normalized_question = re.sub(r"[-_]", " ", question.casefold())
    matches: list[tuple[int, int, str]] = []
    standalone_infrastructure = {"api gateway", "nginx", "postgres"}
    for service in service_catalog:
        normalized_service = re.sub(r"[-_]", " ", service.casefold())
        if normalized_service.endswith(" service"):
            aliases = [normalized_service]
        elif normalized_service in standalone_infrastructure:
            aliases = [normalized_service]
            if normalized_service == "postgres":
                aliases.append("postgresql")
        else:
            continue

        positions: list[int] = []
        for alias in set(aliases):
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            match = re.search(pattern, normalized_question)
            if match:
                positions.append(match.start())
        position = min(positions) if positions else -1
        if position >= 0:
            matches.append((position, -len(normalized_service), service))
    matches.sort()
    return [service for _, _, service in matches]


def resolve_unsupported_service(
    question: str,
    service_catalog: Sequence[str] = DEFAULT_SERVICE_CATALOG,
) -> str | None:
    """Detect an explicitly named service that is outside the catalog."""

    known = {re.sub(r"[-_]", " ", service.casefold()) for service in service_catalog}
    known_aliases = set(known)
    known_aliases.update(
        service.removesuffix(" service")
        for service in known
        if service.endswith(" service")
    )
    for pattern in _UNKNOWN_SERVICE_PATTERNS:
        match = pattern.search(question)
        if not match:
            continue
        candidate = match.group("name").casefold().replace("_", " ").replace("-", " ")
        if candidate in _NON_SERVICE_WORDS or candidate in known_aliases:
            continue
        return candidate
    return None


def _clock_range(match: re.Match[str], anchor: datetime) -> tuple[datetime, datetime] | None:
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    ampm = (match.group("ampm") or match.group("bare_ampm") or "").replace(".", "")

    if minute > 59:
        return None
    if ampm:
        if hour < 1 or hour > 12:
            return None
        if ampm == "am":
            hour = 0 if hour == 12 else hour
        else:
            hour = 12 if hour == 12 else hour + 12
    elif hour > 23:
        return None

    center = _day_start(anchor) + timedelta(hours=hour, minutes=minute)
    return center - timedelta(minutes=30), center + timedelta(minutes=30)


def resolve_time_range(question: str, now: datetime) -> tuple[datetime, datetime] | None:
    """Resolve supported relative time phrases against a fixed UTC corpus anchor."""

    anchor = _utc(now)
    lowered = question.casefold()
    day = _day_start(anchor)

    if "last night" in lowered:
        return day - timedelta(hours=6), day + timedelta(hours=6)
    if "last week" in lowered:
        current_week_start = day - timedelta(days=day.weekday())
        return current_week_start - timedelta(days=7), current_week_start
    if "tonight" in lowered:
        return day + timedelta(hours=18), day + timedelta(days=1, hours=6)
    if "this morning" in lowered:
        return day, day + timedelta(hours=12)
    if "yesterday" in lowered:
        return day - timedelta(days=1), day
    if "today" in lowered:
        return day, day + timedelta(days=1)
    if _MIDNIGHT_RE.search(question):
        center = day
        return center - timedelta(minutes=30), center + timedelta(minutes=30)

    match = _CLOCK_RE.search(question)
    if match:
        resolved = _clock_range(match, anchor)
        if resolved is not None:
            return resolved

    logger.debug("no supported temporal expression found in query: %s", question)
    return None


def resolve_unsupported_time_expression(question: str) -> str | None:
    """Return an explicit temporal phrase not supported by the parser."""

    match = _UNSUPPORTED_TIME_RE.search(question)
    return match.group(0) if match else None


_TOPIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "at",
    "be",
    "did",
    "for",
    "from",
    "happened",
    "how",
    "in",
    "is",
    "me",
    "of",
    "on",
    "show",
    "the",
    "there",
    "to",
    "what",
    "were",
    "why",
    "with",
}


def classify_intent(
    question: str,
    time_range: tuple[datetime, datetime] | None,
    service_filter: Sequence[str],
) -> str:
    if time_range is None:
        return "topic_only"
    if service_filter:
        return "temporal_and_topic"

    remaining = _TEMPORAL_WORD_RE.sub(" ", question.casefold())
    tokens = re.findall(r"[a-z0-9]+", remaining)
    has_topic = any(token not in _TOPIC_STOPWORDS for token in tokens)
    return "temporal_and_topic" if has_topic else "temporal_only"


def parse_query(
    question: str,
    now: datetime,
    service_catalog: Sequence[str] = DEFAULT_SERVICE_CATALOG,
) -> ParsedQuery:
    if not question.strip():
        raise ValueError("question must not be blank")
    time_range = resolve_time_range(question, now)
    service_filter = resolve_service_filter(question, service_catalog)
    unsupported_service = resolve_unsupported_service(question, service_catalog)
    unsupported_time_expression = (
        None
        if time_range is not None
        else resolve_unsupported_time_expression(question)
    )
    return ParsedQuery(
        original=question,
        intent=classify_intent(
            question,
            time_range,
            service_filter,
        ),
        time_range=time_range,
        service_filter=service_filter,
        unsupported_service=unsupported_service,
        unsupported_time_expression=unsupported_time_expression,
    )
