"""Incident scenario definitions."""

from scenarios.checkout_502 import CHECKOUT_502_SCENARIO
from scenarios.notification_rate_limit import NOTIFICATION_RATE_LIMIT_SCENARIO
from scenarios.payment_timeout import PAYMENT_TIMEOUT_SCENARIO
from scenarios.postgres_lock_contention import POSTGRES_LOCK_CONTENTION_SCENARIO


SCENARIOS = {
    scenario.scenario_id: scenario
    for scenario in (
        PAYMENT_TIMEOUT_SCENARIO,
        CHECKOUT_502_SCENARIO,
        POSTGRES_LOCK_CONTENTION_SCENARIO,
        NOTIFICATION_RATE_LIMIT_SCENARIO,
    )
}

__all__ = ["SCENARIOS"]
