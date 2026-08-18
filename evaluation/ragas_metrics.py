"""RAGAS metric setup using Groq/Ollama-compatible LLMs and local BGE embeddings."""

from __future__ import annotations

import asyncio
from functools import lru_cache
import os
from typing import Any

try:
    from ragas.embeddings.base import BaseRagasEmbedding
except ImportError:  # Keep question/abstention checks importable without eval extras.
    class BaseRagasEmbedding:  # type: ignore[no-redef]
        pass


class SentenceTransformerRagasEmbeddings(BaseRagasEmbedding):
    """Minimal RAGAS-compatible embedding adapter.

    It intentionally uses the same model and normalization settings as the
    production retriever, so answer relevancy is measured in the same embedding
    space as retrieval.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        super().__init__()
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str, **_: Any) -> list[float]:
        vector = self.model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text, **kwargs)

    def embed_texts(self, texts: list[str], **_: Any) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_texts, texts, **kwargs)

    # These aliases also allow the object to work with RAGAS versions that use
    # the legacy LangChain-style embedding method names.
    embed_query = embed_text
    embed_documents = embed_texts
    aembed_query = aembed_text
    aembed_documents = aembed_texts


@lru_cache(maxsize=1)
def build_ragas_components(
    provider: str,
    model_name: str,
    embedding_model: str,
    ollama_base_url: str,
    groq_base_url: str,
    groq_api_key: str | None,
) -> tuple[Any, Any]:
    """Build the evaluator LLM and embeddings once per harness process."""

    from ragas.llms import llm_factory

    if provider == "groq":
        from openai import AsyncOpenAI

        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required for RAGAS with Groq")
        client = AsyncOpenAI(api_key=groq_api_key, base_url=groq_base_url)
    elif provider == "ollama":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key="ollama",
            base_url=f"{ollama_base_url.rstrip('/')}/v1",
        )
    else:
        raise RuntimeError(f"unsupported evaluator provider: {provider!r}")

    ragas_max_tokens = int(os.getenv("RAGAS_MAX_TOKENS", "2048"))
    llm = llm_factory(
        model_name,
        client=client,
        temperature=0,
        max_tokens=ragas_max_tokens,
        reasoning_effort="low",
    )
    embeddings = SentenceTransformerRagasEmbeddings(embedding_model)
    return llm, embeddings


async def score_ragas(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str | None,
    provider: str,
    model_name: str,
    embedding_model: str,
    ollama_base_url: str,
    groq_base_url: str,
    groq_api_key: str | None,
    metric_delay_seconds: float = 0.0,
) -> dict[str, float | None]:
    """Score one answer. Reference-dependent metrics remain null without a reference."""

    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    llm, embeddings = build_ragas_components(
        provider,
        model_name,
        embedding_model,
        ollama_base_url,
        groq_base_url,
        groq_api_key,
    )
    scores: dict[str, float | None] = {
        "faithfulness": None,
        "context_precision": None,
        "context_recall": None,
        "answer_relevancy": None,
    }

    async def run_metric(name: str, metric: Any, **kwargs: Any) -> None:
        try:
            if metric_delay_seconds > 0:
                print(
                    f"waiting {metric_delay_seconds:.1f}s before RAGAS metric {name}...",
                    flush=True,
                )
                await asyncio.sleep(metric_delay_seconds)
            result = await metric.ascore(**kwargs)
            scores[name] = float(result.value)
        except Exception as exc:
            # One metric failing should not discard the API result or the other scores.
            scores[name] = None
            print(f"RAGAS metric {name} failed: {exc}")

    await run_metric(
        "faithfulness",
        Faithfulness(llm=llm),
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    await run_metric(
        "answer_relevancy",
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        user_input=question,
        response=answer,
    )

    if reference:
        from ragas.metrics.collections import ContextPrecision, ContextRecall

        await run_metric(
            "context_precision",
            ContextPrecision(llm=llm),
            user_input=question,
            reference=reference,
            retrieved_contexts=contexts,
        )
        await run_metric(
            "context_recall",
            ContextRecall(llm=llm),
            user_input=question,
            reference=reference,
            retrieved_contexts=contexts,
        )

    return scores
