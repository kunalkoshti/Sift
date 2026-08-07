"""Payment-timeout incident scenario and its separate ground truth."""

from __future__ import annotations

from log_generator.scenarios.base import GroundTruth, ScenarioDefinition, ScenarioEvent


PAYMENT_TIMEOUT_SCENARIO = ScenarioDefinition(
    scenario_id="payment-timeout-v1",
    trace_id="trace-payment-timeout-001",
    events=(
        ScenarioEvent(
            event_id="payment-memory-growth",
            service="payment-service",
            level="WARN",
            message=(
                "Payment worker memory usage increased to 768MB after 35 minutes; "
                "possible memory leak detected"
            ),
            relative_offset_seconds=-120,
            metadata={"memory_mb": 768, "uptime_minutes": 35, "percent": 75},
        ),
        ScenarioEvent(
            event_id="payment-heap-pressure",
            service="payment-service",
            level="WARN",
            message="Payment worker heap utilization reached 94%",
            relative_offset_seconds=-90,
            metadata={"percent": 94},
        ),
        ScenarioEvent(
            event_id="payment-oom-kill",
            service="payment-service",
            level="CRITICAL",
            message="Payment worker killed by OOM after memory usage reached 1024MB",
            relative_offset_seconds=-60,
            metadata={"memory_mb": 1024, "memory_limit_mb": 1024},
        ),
        ScenarioEvent(
            event_id="payment-auto-restart",
            service="payment-service",
            level="CRITICAL",
            message="Payment service auto-restarted after process exit (restart 1)",
            relative_offset_seconds=-45,
            metadata={"restart_count": 1, "restart_reason": "oom_kill"},
        ),
        ScenarioEvent(
            event_id="payment-readiness-failure",
            service="payment-service",
            level="CRITICAL",
            message="Payment service readiness probe failed after auto-restart",
            relative_offset_seconds=-30,
            metadata={"probe": "readiness", "restart_count": 1},
        ),
        ScenarioEvent(
            event_id="postgres-pool-exhausted",
            service="postgres",
            level="ERROR",
            message="Database connection pool exhausted for payment-service",
            relative_offset_seconds=-15,
            metadata={
                "client_service": "payment-service",
                "pool_size": 50,
                "active_connections": 50,
            },
        ),
        ScenarioEvent(
            event_id="payment-authorization-timeout",
            service="payment-service",
            level="ERROR",
            message="Payment authorization timed out for order order-84721",
            relative_offset_seconds=0,
            metadata={"order_id": "order-84721", "payment_id": "payment-84721"},
        ),
        ScenarioEvent(
            event_id="checkout-payment-failure",
            service="checkout-service",
            level="ERROR",
            message="Unable to advance order order-84721 to payment authorization",
            relative_offset_seconds=20,
            metadata={"order_id": "order-84721", "checkout_id": "checkout-84721"},
        ),
    ),
    ground_truth=GroundTruth(
        scenario_id="payment-timeout-v1",
        root_cause_description=(
            "A memory leak in payment-service caused sustained heap growth and an OOM kill. "
            "The automatic restart failed readiness checks, and the resulting recovery pressure "
            "exhausted the PostgreSQL connection pool."
        ),
        expected_answer=(
            "The payment timeout was caused by a payment-service memory leak that triggered an "
            "OOM restart and left the service unable to obtain PostgreSQL connections."
        ),
        smoking_gun_event_ids=(
            "payment-memory-growth",
            "payment-oom-kill",
            "payment-auto-restart",
            "postgres-pool-exhausted",
        ),
    ),
)
