"""RAGAS metric setup using Groq/Ollama-compatible LLMs and local BGE embeddings."""

from __future__ import annotations

import asyncio
from functools import lru_cache
import os
from typing import Any

from evaluation.provider import EvaluationLLMConfig

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
    config: EvaluationLLMConfig,
    embedding_model: str,
) -> tuple[Any, Any]:
    """Build the configured evaluator LLM and local embeddings once per process."""

    from ragas.llms import llm_factory

    client = config.build_client()

    ragas_max_tokens = int(os.getenv("RAGAS_MAX_TOKENS", "2048"))
    llm_options: dict[str, Any] = {
        "client": client,
        "temperature": 0,
        "max_tokens": ragas_max_tokens,
    }
    llm_options.update(config.request_options())
    llm = llm_factory(
        config.model,
        **llm_options,
    )
    embeddings = SentenceTransformerRagasEmbeddings(embedding_model)
    return llm, embeddings


async def score_ragas(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str | None,
    config: EvaluationLLMConfig,
    embedding_model: str,
    metric_delay_seconds: float = 0.0,
) -> dict[str, float | None]:
    """Score one answer. Reference-dependent metrics remain null without a reference."""

    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    llm, embeddings = build_ragas_components(
        config,
        embedding_model,
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
