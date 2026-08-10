"""Minimal LangChain retrieve-then-stuff QA chain."""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from log_api.retriever import RetrievedChunk


SYSTEM_PROMPT = """You answer questions about application logs.
Use only the supplied log context. Do not invent events, causes, timestamps, or services.
Read events in chronological order, even when they come from multiple retrieved chunks.
When the logs support a causal chain, identify the initiating event, intermediate
failures, and final customer-visible symptom. Prefer related non-null trace IDs and
treat chunks with empty trace_ids as background noise. Say that the logs are
insufficient only when the evidence genuinely does not support a conclusion.
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
        )

    raise RuntimeError(
        f"unsupported LLM_PROVIDER={provider!r}; choose 'groq' or 'ollama'"
    )


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No matching log chunks were retrieved."

    sections = []
    for index, chunk in enumerate(chunks, start=1):
        sections.append(
            f"[Chunk {index}; id={chunk.id}; window={chunk.window_start.isoformat()}"
            f"..{chunk.window_end.isoformat()}; services={','.join(chunk.services)}; "
            f"trace_ids={','.join(chunk.trace_ids) if chunk.trace_ids else 'noise'}]\n"
            f"{chunk.content}"
        )
    return "\n\n---\n\n".join(sections)


class QAChain:
    def __init__(
        self,
        provider: str,
        model_name: str,
        ollama_base_url: str,
        groq_base_url: str,
        groq_api_key: str | None,
    ):
        self.chain = PROMPT | build_llm(
            provider,
            model_name,
            ollama_base_url,
            groq_base_url,
            groq_api_key,
        ) | StrOutputParser()

    async def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        return await self.chain.ainvoke(
            {
                "question": question,
                "context": format_context(chunks),
            }
        )
