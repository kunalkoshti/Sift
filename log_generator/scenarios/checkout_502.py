"""Checkout-service crash-loop incident that produces nginx 502 responses."""

from scenarios.base import GroundTruth, ScenarioDefinition, ScenarioEvent


CHECKOUT_502_SCENARIO = ScenarioDefinition(
    scenario_id="checkout-502-v1",
    trace_id="trace-checkout-502-001",
    events=(
        ScenarioEvent(
            event_id="checkout-config-validation-failed",
            service="checkout-service",
            level="CRITICAL",
            message="Checkout service configuration validation failed during startup",
            relative_offset_seconds=-90,
            metadata={
                "deployment_id": "checkout-2026-08-07.3",
                "config_key": "PAYMENT_TIMEOUT_MS",
            },
        ),
        ScenarioEvent(
            event_id="checkout-readiness-failure",
            service="checkout-service",
            level="CRITICAL",
            message="Checkout service readiness probe failed after configuration rollout",
            relative_offset_seconds=-75,
            metadata={"probe": "readiness", "deployment_id": "checkout-2026-08-07.3"},
        ),
        ScenarioEvent(
            event_id="checkout-crash-loop",
            service="checkout-service",
            level="CRITICAL",
            message="Checkout service entered crash loop after 3 restart attempts",
            relative_offset_seconds=-60,
            metadata={"restart_count": 3, "deployment_id": "checkout-2026-08-07.3"},
        ),
        ScenarioEvent(
            event_id="nginx-upstream-timeout",
            service="nginx",
            level="WARN",
            message="Upstream response for /checkout/order-93612 exceeded 3000ms",
            relative_offset_seconds=-30,
            metadata={"path": "/checkout/order-93612", "duration_ms": 3000},
        ),
        ScenarioEvent(
            event_id="nginx-502",
            service="nginx",
            level="ERROR",
            message="Returned gateway error status 502 for /checkout/order-93612",
            relative_offset_seconds=-10,
            metadata={
                "path": "/checkout/order-93612",
                "status": 502,
                "upstream_service": "checkout-service",
            },
        ),
        ScenarioEvent(
            event_id="gateway-checkout-failure",
            service="api-gateway",
            level="ERROR",
            message="Upstream request failed with status 502",
            relative_offset_seconds=10,
            metadata={
                "method": "POST",
                "path": "/checkout/order-93612",
                "status": 502,
            },
        ),
    ),
    ground_truth=GroundTruth(
        scenario_id="checkout-502-v1",
        root_cause_description=(
            "A checkout-service configuration error caused startup validation to fail. "
            "The deployment entered a crash loop, leaving nginx without a healthy upstream."
        ),
        expected_answer=(
            "The 502 responses were caused by the checkout-service crash loop after the "
            "configuration rollout, not by an nginx failure."
        ),
        smoking_gun_event_ids=(
            "checkout-config-validation-failed",
            "checkout-readiness-failure",
            "checkout-crash-loop",
        ),
    ),
)
