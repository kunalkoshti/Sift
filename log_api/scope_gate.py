"""Deterministic scope and answerability checks for the third self-RAG gate."""

from __future__ import annotations

from dataclasses import dataclass
import re


VALID_SCOPE_GATE_MODES = {"off", "shadow", "enforce"}


_OUT_OF_SCOPE_RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "external_state",
        "weather and other external state are not present in application logs",
        re.compile(
            r"\b(?:weather|forecast|temperature|rainfall|climate)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "financial_metric",
        "financial metrics are not present in application logs",
        re.compile(
            r"\b(?:company|business|organization)'?s?\s+"
            r"(?:revenue|sales|profit|turnover)\b|"
            r"\b(?:revenue|sales|profit|turnover)\b.*\b"
            r"(?:quarter|year|month|last|this|previous)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "customer_demographic",
        "customer segments and demographics are not present in application logs",
        re.compile(
            r"\b(?:customer|user)\s+(?:segment|segments|demographic|demographics)\b|"
            r"\bcustomer\s+churn\b|"
            r"\b(?:age|gender|ethnicity|income)\s+(?:group|segment|distribution)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "aggregate_business_statistic",
        "aggregate business statistics are not supported by this log corpus",
        re.compile(
            r"\bwhat\s+percentage\b|"
            r"\bpercentage\s+of\b|"
            r"\b(?:payment|checkout|order|transaction)\s+failure\s+rate(?!-)\b|"
            r"\b(?:payment|checkout|order|transaction)\s+success\s+rate(?!-)\b|"
            r"\bwhich\s+customers?\s+.*\bmost\s+(?:often|frequently)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class ScopeGateDecision:
    mode: str
    in_scope: bool
    eligible: bool
    would_abstain: bool
    category: str | None
    reason: str
    note: str | None = None

    @property
    def enforced_abstention(self) -> bool:
        return self.mode == "enforce" and self.would_abstain


def evaluate_scope_gate(question: str, *, mode: str) -> ScopeGateDecision:
    """Classify only clearly unsupported question categories.

    The gate intentionally uses high-precision rules. Questions that do not
    match a clear rule continue through normal retrieval and later gates.
    """

    normalized_mode = mode.casefold()
    if normalized_mode not in VALID_SCOPE_GATE_MODES:
        raise ValueError(
            "scope gate mode must be one of: "
            + ", ".join(sorted(VALID_SCOPE_GATE_MODES))
        )
    if not question.strip():
        raise ValueError("scope gate question must not be blank")

    if normalized_mode == "off":
        return ScopeGateDecision(
            mode=normalized_mode,
            in_scope=True,
            eligible=False,
            would_abstain=False,
            category=None,
            reason="gate_disabled",
        )

    for category, explanation, pattern in _OUT_OF_SCOPE_RULES:
        if pattern.search(question):
            return ScopeGateDecision(
                mode=normalized_mode,
                in_scope=False,
                eligible=True,
                would_abstain=True,
                category=category,
                reason=explanation,
                note=f"Scope gate classified the question as {category}.",
            )

    return ScopeGateDecision(
        mode=normalized_mode,
        in_scope=True,
        eligible=True,
        would_abstain=False,
        category=None,
        reason="no_clear_scope_boundary_detected",
    )


def deterministic_scope_answer(decision: ScopeGateDecision) -> str:
    """Return a safe answer for a clearly out-of-scope question."""

    if decision.in_scope or not decision.category:
        raise ValueError("deterministic scope answer requires an out-of-scope decision")
    return (
        "This log corpus cannot answer that question reliably because "
        f"{decision.reason}."
    )
