"""PostgreSQL lock-contention incident causing checkout and payment failures."""

from scenarios.base import GroundTruth, ScenarioDefinition, ScenarioEvent


POSTGRES_LOCK_CONTENTION_SCENARIO = ScenarioDefinition(
    scenario_id="postgres-lock-contention-v1",
    trace_id="trace-postgres-lock-001",
    events=(
        ScenarioEvent(
            event_id="orders-migration-lock",
            service="postgres",
            level="CRITICAL",
            message="Migration transaction acquired an exclusive lock on orders",
            relative_offset_seconds=-120,
            metadata={
                "relation": "orders",
                "lock_mode": "ACCESS EXCLUSIVE",
                "migration_id": "orders-2026-08-07.2",
            },
        ),
        ScenarioEvent(
            event_id="orders-lock-waiters",
            service="postgres",
            level="WARN",
            message="Lock waiters reached 18 for relation orders",
            relative_offset_seconds=-90,
            metadata={"relation": "orders", "waiter_count": 18},
        ),
        ScenarioEvent(
            event_id="order-query-latency",
            service="postgres",
            level="WARN",
            message="order_lookup query exceeded 2800ms while waiting for a lock",
            relative_offset_seconds=-60,
            metadata={
                "query_name": "order_lookup",
                "duration_ms": 2800,
                "relation": "orders",
            },
        ),
        ScenarioEvent(
            event_id="order-query-lock-timeout",
            service="postgres",
            level="ERROR",
            message="order_lookup query aborted after lock timeout",
            relative_offset_seconds=-30,
            metadata={
                "query_name": "order_lookup",
                "lock_timeout_ms": 3000,
                "relation": "orders",
            },
        ),
        ScenarioEvent(
            event_id="checkout-order-timeout",
            service="checkout-service",
            level="ERROR",
            message="Checkout request failed for order order-41209",
            relative_offset_seconds=-5,
            metadata={"order_id": "order-41209", "dependency": "postgres"},
        ),
        ScenarioEvent(
            event_id="payment-ledger-failure",
            service="payment-service",
            level="ERROR",
            message="Failed to record payment ledger entry for order order-41209",
            relative_offset_seconds=15,
            metadata={"order_id": "order-41209", "dependency": "postgres"},
        ),
        ScenarioEvent(
            event_id="gateway-504",
            service="api-gateway",
            level="ERROR",
            message="Upstream request failed with status 504",
            relative_offset_seconds=35,
            metadata={
                "method": "POST",
                "path": "/checkout/order-41209",
                "status": 504,
            },
        ),
    ),
    ground_truth=GroundTruth(
        scenario_id="postgres-lock-contention-v1",
        root_cause_description=(
            "A migration held an exclusive lock on the orders table. Growing lock waiters "
            "caused order lookups to time out, which propagated into checkout and payment failures."
        ),
        expected_answer=(
            "The checkout and payment failures were caused by PostgreSQL lock contention from "
            "the orders migration, resulting in query timeouts and downstream 504s."
        ),
        smoking_gun_event_ids=(
            "orders-migration-lock",
            "orders-lock-waiters",
            "order-query-lock-timeout",
        ),
    ),
)
