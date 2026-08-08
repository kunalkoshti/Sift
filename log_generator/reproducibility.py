"""Shared randomness and clock policy for deterministic corpus generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import random

from dotenv import load_dotenv


ROOT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ROOT_ENV_FILE)


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
        configured_timestamp = os.getenv("GENERATOR_BASE_TIMESTAMP")
        if not configured_timestamp:
            raise RuntimeError(
                "GENERATOR_BASE_TIMESTAMP must be configured for seeded generation"
            )
        base_timestamp = datetime.fromisoformat(configured_timestamp)
    else:
        base_timestamp = now or datetime.now(timezone.utc)

    if base_timestamp.tzinfo is None or base_timestamp.utcoffset() is None:
        raise ValueError("base timestamp must be timezone-aware")

    return GenerationContext(
        seed=seed,
        rng=random.Random(seed),
        base_timestamp=base_timestamp.astimezone(timezone.utc),
    )
