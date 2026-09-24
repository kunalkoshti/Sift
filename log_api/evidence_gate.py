"""Post-generation evidence verification for the second self-RAG gate."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from log_api.qa import build_llm, format_context
from log_api.retriever import RetrievedChunk


EVIDENCE_SYSTEM_PROMPT = """You are an evidence verifier for an application-log QA system.
Judge only whether the answer's factual claims are supported by the supplied
retrieved log context. Do not use general world knowledge and do not infer that
an event happened merely because it would be plausible.

Return exactly one JSON object with this shape and no Markdown:
{{
  "supported": true or false,
  "unsupported_claims": ["claim text"],
  "evidence_references": [
    {{
      "claim": "claim text",
      "chunk_id": "UUID or identifier",
      "timestamp": "timestamp from the log context",
      "service": "service name from the log context"
    }}
  ],
  "reason": "short explanation"
}}

Mark supported=false when one or more important factual claims are not supported
by a specific log line. A statement about uncertainty or lack of evidence is
supported when the supplied context and retrieval note justify that statement.
Every factual claim in an answer must have an evidence reference. A causal claim
is not directly supported merely because one event appears before another; if the
logs show sequence but not causation, mark the causal claim unsupported or require
the answer to use cautious wording such as "preceded" or "is consistent with".
Only cite chunk IDs, timestamps, and services that appear in the supplied context.
Do not mark an answer unsupported merely because it is concise or because it
contains no factual claim beyond an explicitly supported uncertainty statement.
"""


EVIDENCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", EVIDENCE_SYSTEM_PROMPT),
        (
            "human",
            "Question:\n{question}\n\nAnswer to verify:\n{answer}\n\n"
            "Retrieved log context:\n{context}",
        ),
    ]
)


@dataclass(frozen=True)
class EvidenceReference:
    claim: str
    chunk_id: str
    timestamp: str
    service: str


@dataclass(frozen=True)
class EvidenceVerification:
    supported: bool
    unsupported_claims: tuple[str, ...]
    evidence_references: tuple[EvidenceReference, ...]
    reason: str
    raw_output: str

    @property
    def unsupported_claim_count(self) -> int:
        return len(self.unsupported_claims)

    def requires_action(self, max_unsupported_claims: int) -> bool:
        if max_unsupported_claims < 0:
            raise ValueError("max_unsupported_claims must be non-negative")
        return (
            not self.supported
            or self.unsupported_claim_count > max_unsupported_claims
        )


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"evidence verifier field {field_name!r} must be a non-empty string")
    return value.strip()


def _parse_json_object(raw_output: str) -> dict[str, Any]:
    cleaned = raw_output.strip()
    if not cleaned:
        raise ValueError("evidence verifier returned empty output")

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    if start < 0:
        raise ValueError("evidence verifier output did not contain a JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("evidence verifier returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("evidence verifier JSON must be an object")
    return value


def parse_evidence_verification(raw_output: str) -> EvidenceVerification:
    """Parse and validate the verifier's single JSON response."""

    payload = _parse_json_object(raw_output)
    supported = payload.get("supported")
    if not isinstance(supported, bool):
        raise ValueError("evidence verifier field 'supported' must be boolean")

    unsupported = payload.get("unsupported_claims", [])
    if not isinstance(unsupported, list) or not all(
        isinstance(item, str) and item.strip() for item in unsupported
    ):
        raise ValueError("evidence verifier field 'unsupported_claims' must be a string list")
    unsupported_claims = tuple(item.strip() for item in unsupported)

    references = payload.get("evidence_references", [])
    if not isinstance(references, list):
        raise ValueError("evidence verifier field 'evidence_references' must be a list")
    evidence_references: list[EvidenceReference] = []
    for reference in references:
        if not isinstance(reference, dict):
            raise ValueError("each evidence reference must be an object")
        evidence_references.append(
            EvidenceReference(
                claim=_required_string(reference.get("claim"), "claim"),
                chunk_id=_required_string(reference.get("chunk_id"), "chunk_id"),
                timestamp=_required_string(reference.get("timestamp"), "timestamp"),
                service=_required_string(reference.get("service"), "service"),
            )
        )

    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("evidence verifier field 'reason' must be a string")

    return EvidenceVerification(
        supported=supported,
        unsupported_claims=unsupported_claims,
        evidence_references=tuple(evidence_references),
        reason=reason.strip(),
        raw_output=raw_output,
    )


def _timestamp_matches_chunk(timestamp: str, chunk: RetrievedChunk) -> bool:
    """Check that a verifier timestamp belongs to the referenced chunk."""

    candidate = timestamp.strip()
    if candidate in chunk.content:
        return True

    # The embedded content uses HH:MM:SS while a verifier may return an ISO
    # timestamp copied from the chunk header. Accept the time portion when it
    # is present in the chunk's log lines.
    if "T" in candidate:
        time_part = candidate.split("T", 1)[1].split(".", 1)[0]
        time_part = time_part.rstrip("Z").split("+", 1)[0]
        if time_part in chunk.content:
            return True

    return candidate in {
        chunk.window_start.isoformat(),
        chunk.window_end.isoformat(),
    }


def validate_evidence_verification(
    verification: EvidenceVerification,
    chunks: list[RetrievedChunk],
    answer: str,
) -> EvidenceVerification:
    """Validate verifier references against the actual retrieved chunks.

    The LLM verifier decides whether a claim is semantically supported. This
    deterministic layer makes sure its references are real references into the
    context supplied to it, rather than merely plausible-looking identifiers.
    """

    chunks_by_id = {str(chunk.id): chunk for chunk in chunks}
    invalid_claims: list[str] = []

    for reference in verification.evidence_references:
        chunk = chunks_by_id.get(reference.chunk_id)
        if chunk is None:
            invalid_claims.append(
                f"{reference.claim} (references unknown chunk {reference.chunk_id})"
            )
            continue
        if reference.service not in chunk.services:
            invalid_claims.append(
                f"{reference.claim} (service {reference.service!r} is not in chunk "
                f"{reference.chunk_id})"
            )
            continue
        if not _timestamp_matches_chunk(reference.timestamp, chunk):
            invalid_claims.append(
                f"{reference.claim} (timestamp {reference.timestamp!r} is not in "
                f"chunk {reference.chunk_id})"
            )

    # A non-empty answer with factual content must contain at least one
    # verifiable reference. Deterministic empty-context answers do not reach
    # the evidence gate, so this safely catches unsupported LLM answers here.
    if verification.supported and answer.strip() and not verification.evidence_references:
        invalid_claims.append(
            "The answer contains no verifiable evidence reference for its factual claims."
        )

    if not invalid_claims:
        return verification

    merged_claims = tuple(dict.fromkeys((*verification.unsupported_claims, *invalid_claims)))
    reason = verification.reason.strip()
    validation_reason = "At least one claim failed deterministic evidence-reference validation."
    reason = f"{reason} {validation_reason}".strip()
    return replace(
        verification,
        supported=False,
        unsupported_claims=merged_claims,
        reason=reason,
    )


class EvidenceVerifier:
    """Call the verifier model and validate its structured result."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        ollama_base_url: str,
        groq_base_url: str,
        groq_api_key: str | None,
        max_context_chars: int,
    ):
        self.chain = EVIDENCE_PROMPT | build_llm(
            provider,
            model_name,
            ollama_base_url,
            groq_base_url,
            groq_api_key,
        ) | StrOutputParser()
        self.max_context_chars = max_context_chars

    async def verify(
        self,
        question: str,
        answer: str,
        chunks: list[RetrievedChunk],
        retrieval_note: str | None = None,
    ) -> EvidenceVerification:
        raw_output = await self.chain.ainvoke(
            {
                "question": question,
                "answer": answer,
                "context": format_context(
                    chunks,
                    retrieval_note,
                    max_chars=self.max_context_chars,
                ),
            }
        )
        verification = parse_evidence_verification(raw_output)
        return validate_evidence_verification(verification, chunks, answer)
