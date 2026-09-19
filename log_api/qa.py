"""Minimal LangChain retrieve-then-stuff QA chain."""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from log_api.retriever import RetrievedChunk


SYSTEM_PROMPT = """You answer questions about application logs.
Use only the supplied log context. Do not invent events, causes, timestamps, or services.
Read events in chronological order, even when they come from multiple retrieved chunks.
When multiple TRACE sections are present, analyze each trace separately and do not
merge events from different traces into one causal chain. If the question asks for
one cause but multiple traces are plausible, state the competing incidents and
the uncertainty explicitly.
If the retrieval note says multiple incident traces were found, preserve that
ambiguity: compare the traces separately and do not claim that one is the unique
cause unless the log evidence clearly rules out the others.
When only one trace is present, describe what that retrieved trace supports. Do
not treat the absence of another trace from the retrieved context as proof that
the other incident does not exist in the corpus. Use scoped wording such as
"the retrieved logs support" or "no evidence was found in the retrieved context"
when an alternative was not retrieved.
When the logs support a causal chain, identify the initiating event, intermediate
failures, and final customer-visible symptom. Prefer related non-null trace IDs and
treat chunks with empty trace_ids as background noise. Say that the logs are
insufficient only when the evidence genuinely does not support a conclusion.
If the retrieval note says a service or temporal expression is unsupported, abstain
and do not substitute evidence from another service or time period.
If the retrieval note says no chunks were found in the requested time range and
fallback evidence is supplied, explicitly disclose that the requested range was
empty and that the answer uses evidence from the available corpus or another time
range.
Be concise, distinguish facts from uncertainty, and cite the relevant chunk number
when explaining the conclusion.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Question:\n{question}\n\nRetrieved log context:\n{context}",
        ),
    ]
)


def deterministic_empty_context_answer(note: str | None = None) -> str:
    """Return a safe answer without asking the LLM to reason over no evidence."""

    normalized_note = (note or "").casefold()
    if (
        "no correlated incident" in normalized_note
        or "no chunks were found for the requested service" in normalized_note
    ):
        return (
            "No trace-correlated incident was found for the requested service "
            "in the available log data."
        )
    return "The available logs contain no evidence that can answer this question."


def build_llm(
    provider: str,
    model_name: str,
    ollama_base_url: str,
    groq_base_url: str,
    groq_api_key: str | None,
) -> Any:
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model_name, base_url=ollama_base_url, temperature=0)

    if provider == "groq":
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY must be configured when LLM_PROVIDER='groq'")

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=groq_api_key,
            base_url=groq_base_url,
            temperature=0,
            reasoning_effort="low",
        )

    raise RuntimeError(
        f"unsupported LLM_PROVIDER={provider!r}; choose 'groq' or 'ollama'"
    )


def _cap_context_text(context: str, max_chars: int | None) -> str:
    if max_chars is None:
        return context
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if len(context) <= max_chars:
        return context

    marker = "\n\n[CONTEXT TRUNCATED TO FIT THE MODEL INPUT LIMIT]\n\n"
    if max_chars <= len(marker):
        return marker[:max_chars]
    available = max_chars - len(marker)
    head_chars = (available + 1) // 2
    tail_chars = available - head_chars
    tail = context[-tail_chars:] if tail_chars else ""
    return context[:head_chars] + marker + tail


def format_context(
    chunks: list[RetrievedChunk],
    note: str | None = None,
    max_chars: int | None = None,
) -> str:
    if not chunks:
        context = "No matching log chunks were retrieved."
        formatted = f"RETRIEVAL NOTE: {note}\n\n{context}" if note else context
        return _cap_context_text(formatted, max_chars)

    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: (chunk.window_start, chunk.sub_index, str(chunk.id)),
    )
    grouped: dict[str, list[RetrievedChunk]] = {}
    for chunk in ordered_chunks:
        group_key = ",".join(sorted(chunk.trace_ids)) if chunk.trace_ids else "noise"
        grouped.setdefault(group_key, []).append(chunk)

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (item[1][0].window_start, item[0]),
    )
    trace_sections = []
    chunk_number = 1
    for group_key, group_chunks in ordered_groups:
        group_label = "BACKGROUND NOISE" if group_key == "noise" else f"TRACE {group_key}"
        chunk_sections = [f"=== {group_label} ==="]
        for chunk in group_chunks:
            chunk_sections.append(
                f"[Chunk {chunk_number}; id={chunk.id}; window={chunk.window_start.isoformat()}"
                f"..{chunk.window_end.isoformat()}; services={','.join(chunk.services)}; "
                f"max_level={chunk.max_level or 'unknown'}; "
                f"trace_ids={','.join(chunk.trace_ids) if chunk.trace_ids else 'noise'}]\n"
                f"{chunk.content}"
            )
            chunk_number += 1
        trace_sections.append("\n\n".join(chunk_sections))

    context = "\n\n--- TRACE SEPARATOR ---\n\n".join(trace_sections)
    formatted = f"RETRIEVAL NOTE: {note}\n\n{context}" if note else context
    return _cap_context_text(formatted, max_chars)


class QAChain:
    def __init__(
        self,
        provider: str,
        model_name: str,
        ollama_base_url: str,
        groq_base_url: str,
        groq_api_key: str | None,
        max_context_chars: int = 50000,
    ):
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        self.max_context_chars = max_context_chars
        self.chain = PROMPT | build_llm(
            provider,
            model_name,
            ollama_base_url,
            groq_base_url,
            groq_api_key,
        ) | StrOutputParser()

    async def answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        retrieval_note: str | None = None,
    ) -> str:
        return await self.chain.ainvoke(
            {
                "question": question,
                "context": format_context(
                    chunks,
                    retrieval_note,
                    max_chars=self.max_context_chars,
                ),
            }
        )
