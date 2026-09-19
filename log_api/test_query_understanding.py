from datetime import datetime, timezone

from log_api.query_understanding import (
    classify_intent,
    parse_query,
    resolve_service_filter,
    resolve_time_range,
    resolve_unsupported_service,
    resolve_unsupported_time_expression,
)


ANCHOR = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)


def test_service_filter_matches_known_services_in_query_order():
    assert resolve_service_filter("nginx and payment issues") == ["nginx"]
    assert resolve_service_filter("payment service issues") == ["payment-service"]
    assert resolve_service_filter("checkout failures") == []
    assert resolve_service_filter("checkout service failures") == ["checkout-service"]
    assert resolve_service_filter("Which customers had payment issues?") == []
    assert resolve_service_filter("show me postgres errors") == ["postgres"]
    assert resolve_service_filter("What caused the PostgreSQL lock contention?") == [
        "postgres"
    ]
    assert resolve_service_filter("what happened") == []
    assert resolve_service_filter("show me kafka errors") == []


def test_temporal_filter_resolves_supported_ranges_against_anchor():
    assert resolve_time_range("what happened last night", ANCHOR) == (
        datetime(2025, 12, 31, 18, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 6, tzinfo=timezone.utc),
    )
    assert resolve_time_range("what happened last week", ANCHOR) == (
        datetime(2025, 12, 22, tzinfo=timezone.utc),
        datetime(2025, 12, 29, tzinfo=timezone.utc),
    )
    assert resolve_time_range("around 3am", ANCHOR) == (
        datetime(2026, 1, 1, 2, 30, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 3, 30, tzinfo=timezone.utc),
    )
    assert resolve_time_range("at midnight", ANCHOR) == (
        datetime(2025, 12, 31, 23, 30, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc),
    )
    assert resolve_time_range("what happened", ANCHOR) is None


def test_unsupported_service_is_detected_in_service_questions():
    assert resolve_unsupported_service("Did kafka have any incidents?") == "kafka"
    assert resolve_unsupported_service("What happened in the logs?") is None
    assert resolve_unsupported_service("What happened in postgres?") is None


def test_unsupported_temporal_phrases_are_detected_without_rejecting_supported_ones():
    assert resolve_unsupported_time_expression("what happened two hours ago") == (
        "two hours ago"
    )
    assert resolve_unsupported_time_expression("what happened last week") is None


def test_parse_query_classifies_the_three_intents():
    temporal_only = parse_query("what happened at 3am", ANCHOR)
    assert temporal_only.intent == "temporal_only"
    assert temporal_only.service_filter == []

    temporal_and_topic = parse_query("postgres errors last night", ANCHOR)
    assert temporal_and_topic.intent == "temporal_and_topic"
    assert temporal_and_topic.service_filter == ["postgres"]

    topic_only = parse_query("why did payments fail", ANCHOR)
    assert topic_only.intent == "topic_only"
    assert topic_only.service_filter == []
    assert classify_intent(topic_only.original, None, []) == "topic_only"
