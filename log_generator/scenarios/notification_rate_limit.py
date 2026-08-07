"""Notification-provider rate-limit incident causing delivery backlog."""

from scenarios.base import GroundTruth, ScenarioDefinition, ScenarioEvent


NOTIFICATION_RATE_LIMIT_SCENARIO = ScenarioDefinition(
    scenario_id="notification-rate-limit-v1",
    trace_id="trace-notification-rate-limit-001",
    events=(
        ScenarioEvent(
            event_id="provider-rate-limit-started",
            service="notification-service",
            level="WARN",
            message="Notification provider began returning HTTP 429 responses",
            relative_offset_seconds=-120,
            metadata={"provider": "notify-provider", "status": 429, "channel": "email"},
        ),
        ScenarioEvent(
            event_id="notification-retry-surge",
            service="notification-service",
            level="WARN",
            message="Notification provider retry scheduled for message message-77102",
            relative_offset_seconds=-90,
            metadata={
                "message_id": "message-77102",
                "retry_count": 3,
                "backoff_seconds": 60,
            },
        ),
        ScenarioEvent(
            event_id="notification-queue-growth",
            service="notification-service",
            level="WARN",
            message="Notification queue depth reached 5000",
            relative_offset_seconds=-60,
            metadata={"queue_depth": 5000, "channel": "email"},
        ),
        ScenarioEvent(
            event_id="notification-delivery-failure",
            service="notification-service",
            level="ERROR",
            message="Failed to deliver email notification for order order-55881",
            relative_offset_seconds=-30,
            metadata={
                "order_id": "order-55881",
                "channel": "email",
                "provider_status": 429,
            },
        ),
        ScenarioEvent(
            event_id="notification-dead-letter-threshold",
            service="notification-service",
            level="CRITICAL",
            message="Notification messages exceeded dead-letter queue threshold",
            relative_offset_seconds=0,
            metadata={"dead_letter_count": 1200, "provider": "notify-provider"},
        ),
        ScenarioEvent(
            event_id="notification-provider-recovery-delay",
            service="notification-service",
            level="ERROR",
            message="Notification provider rejected message message-77102",
            relative_offset_seconds=30,
            metadata={"message_id": "message-77102", "provider_status": 429},
        ),
    ),
    ground_truth=GroundTruth(
        scenario_id="notification-rate-limit-v1",
        root_cause_description=(
            "The external notification provider rate-limited the service with HTTP 429 responses. "
            "Retries accumulated faster than they could be delivered, creating a queue and dead-letter backlog."
        ),
        expected_answer=(
            "Notification delivery failures were caused by provider rate limiting, which triggered "
            "retries and exhausted the notification queues."
        ),
        smoking_gun_event_ids=(
            "provider-rate-limit-started",
            "notification-retry-surge",
            "notification-queue-growth",
        ),
    ),
)
