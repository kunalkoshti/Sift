"""Shared randomness and clock policy for deterministic corpus generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import random


# A seeded run must not depend on the machine's current clock.
DETERMINISTIC_BASE_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass
class GenerationContext:
    """State shared by generators during one corpus-generation run."""

    seed: int | None
    rng: random.Random
    base_timestamp: datetime


def create_generation_context(
    seed: int | None = None,
    *,
    now: datetime | None = None,
) -> GenerationContext:
    """Create isolated randomness and a deterministic-or-live base timestamp.

    A supplied seed selects the fixed base timestamp and seeds a new local RNG.
    Without a seed, the base timestamp uses the current UTC clock and the RNG
    is seeded nondeterministically by Python.
    """

    if seed is not None:
        base_timestamp = DETERMINISTIC_BASE_TIMESTAMP
    else:
        base_timestamp = now or datetime.now(timezone.utc)

    if base_timestamp.tzinfo is None or base_timestamp.utcoffset() is None:
        raise ValueError("base timestamp must be timezone-aware")

    return GenerationContext(
        seed=seed,
        rng=random.Random(seed),
        base_timestamp=base_timestamp.astimezone(timezone.utc),
    )
