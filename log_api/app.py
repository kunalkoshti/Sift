"""FastAPI entry point for hybrid retrieval, reranking, and QA."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from log_api.evidence_gate import EvidenceVerifier
from log_api.qa import QAChain, deterministic_empty_context_answer
from log_api.retriever import HybridRetriever, RetrievedChunk, RetrieverConfig
from log_api.scope_gate import (
    deterministic_scope_answer,
    evaluate_scope_gate,
)


ROOT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ROOT_ENV_FILE)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiConfig:
    postgres_dsn: str
    embedding_model: str
    top_k: int
    retrieval_mode: str
    retrieval_candidate_k: int
    trace_candidate_k: int
    reranker_model: str
    rrf_k: int
    dense_weight: float
    lexical_weight: float
    trace_rerank_score_gap: float
    trace_cosine_score_gap: float
    temporal_top_k: int
    max_trace_chunks: int
    confidence_gate_mode: str
    min_dense_similarity: float
    evidence_gate_mode: str
    evidence_model: str | None
    evidence_max_unsupported_claims: int
    evidence_max_retries: int
    scope_gate_mode: str
    max_context_chars: int
    llm_provider: str
    llm_model: str
    ollama_base_url: str
    groq_base_url: str
    groq_api_key: str | None
    log_level: str

    @classmethod
    def from_env(cls) -> ApiConfig:
        def required(name: str) -> str:
            value = os.getenv(name)
            if not value:
                raise RuntimeError(f"{name} must be configured in .env or the environment")
            return value

        top_k = int(required("RETRIEVAL_TOP_K"))
        if top_k <= 0:
            raise RuntimeError("RETRIEVAL_TOP_K must be positive")
        retrieval_mode = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
        if retrieval_mode not in {"dense", "hybrid"}:
            raise RuntimeError("RETRIEVAL_MODE must be either 'dense' or 'hybrid'")
        retrieval_candidate_k = int(os.getenv("RETRIEVAL_CANDIDATE_K", "20"))
        if retrieval_candidate_k <= 0:
            raise RuntimeError("RETRIEVAL_CANDIDATE_K must be positive")
        trace_candidate_k = int(
            os.getenv("TRACE_CANDIDATE_K", str(retrieval_candidate_k))
        )
        if trace_candidate_k <= 0:
            raise RuntimeError("TRACE_CANDIDATE_K must be positive")
        trace_candidate_k = max(trace_candidate_k, top_k)
        rrf_k = int(os.getenv("RRF_K", "60"))
        if rrf_k <= 0:
            raise RuntimeError("RRF_K must be positive")
        dense_weight = float(os.getenv("DENSE_RRF_WEIGHT", "1.0"))
        lexical_weight = float(os.getenv("LEXICAL_RRF_WEIGHT", "1.0"))
        if dense_weight < 0 or lexical_weight < 0:
            raise RuntimeError("RRF weights must be non-negative")
        if dense_weight == 0 and lexical_weight == 0:
            raise RuntimeError("at least one RRF weight must be positive")
        trace_rerank_score_gap = float(
            os.getenv("TRACE_RERANK_SCORE_GAP", "1.0")
        )
        trace_cosine_score_gap = float(os.getenv("TRACE_COSINE_SCORE_GAP", "0.05"))
        if trace_rerank_score_gap < 0:
            raise RuntimeError("TRACE_RERANK_SCORE_GAP must be non-negative")
        if trace_cosine_score_gap < 0:
            raise RuntimeError("TRACE_COSINE_SCORE_GAP must be non-negative")
        temporal_top_k = int(os.getenv("TEMPORAL_TOP_K", "10"))
        max_trace_chunks = int(os.getenv("MAX_TRACE_CHUNKS", "50"))
        max_context_chars = int(os.getenv("MAX_CONTEXT_CHARS", "50000"))
        confidence_gate_mode = os.getenv(
            "SELFRAG_CONFIDENCE_GATE_MODE", "shadow"
        ).lower()
        if confidence_gate_mode not in {"off", "shadow", "enforce"}:
            raise RuntimeError(
                "SELFRAG_CONFIDENCE_GATE_MODE must be off, shadow, or enforce"
            )
        min_dense_similarity = float(
            os.getenv("SELFRAG_MIN_DENSE_SIMILARITY", "0.55")
        )
        if not -1.0 <= min_dense_similarity <= 1.0:
            raise RuntimeError("SELFRAG_MIN_DENSE_SIMILARITY must be between -1 and 1")
        evidence_gate_mode = os.getenv("SELFRAG_EVIDENCE_GATE_MODE", "shadow").lower()
        if evidence_gate_mode not in {"off", "shadow", "enforce"}:
            raise RuntimeError(
                "SELFRAG_EVIDENCE_GATE_MODE must be off, shadow, or enforce"
            )
        evidence_model = os.getenv("SELFRAG_EVIDENCE_MODEL") or None
        evidence_max_unsupported_claims = int(
            os.getenv("SELFRAG_EVIDENCE_MAX_UNSUPPORTED_CLAIMS", "0")
        )
        evidence_max_retries = int(os.getenv("SELFRAG_EVIDENCE_MAX_RETRIES", "1"))
        if evidence_max_unsupported_claims < 0:
            raise RuntimeError(
                "SELFRAG_EVIDENCE_MAX_UNSUPPORTED_CLAIMS must be non-negative"
            )
        if not 0 <= evidence_max_retries <= 1:
            raise RuntimeError("SELFRAG_EVIDENCE_MAX_RETRIES must be 0 or 1")
        scope_gate_mode = os.getenv("SELFRAG_SCOPE_GATE_MODE", "shadow").lower()
        if scope_gate_mode not in {"off", "shadow", "enforce"}:
            raise RuntimeError(
                "SELFRAG_SCOPE_GATE_MODE must be off, shadow, or enforce"
            )
        if temporal_top_k <= 0:
            raise RuntimeError("TEMPORAL_TOP_K must be positive")
        if max_trace_chunks <= 0:
            raise RuntimeError("MAX_TRACE_CHUNKS must be positive")
        if max_context_chars <= 0:
            raise RuntimeError("MAX_CONTEXT_CHARS must be positive")
        llm_provider = required("LLM_PROVIDER").lower()
        if llm_provider not in {"groq", "ollama"}:
            raise RuntimeError("LLM_PROVIDER must be either 'groq' or 'ollama'")

        groq_api_key = os.getenv("GROQ_API_KEY")
        if llm_provider == "groq" and not groq_api_key:
            raise RuntimeError("GROQ_API_KEY must be configured when LLM_PROVIDER='groq'")
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if log_level not in logging.getLevelNamesMapping():
            raise RuntimeError(f"LOG_LEVEL must be a valid logging level, got {log_level!r}")

        return cls(
            postgres_dsn=required("POSTGRES_DSN"),
            embedding_model=required("EMBEDDING_MODEL"),
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            retrieval_candidate_k=retrieval_candidate_k,
            trace_candidate_k=trace_candidate_k,
            reranker_model=os.getenv(
                "RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
            trace_rerank_score_gap=trace_rerank_score_gap,
            trace_cosine_score_gap=trace_cosine_score_gap,
            temporal_top_k=temporal_top_k,
            max_trace_chunks=max_trace_chunks,
            max_context_chars=max_context_chars,
            confidence_gate_mode=confidence_gate_mode,
            min_dense_similarity=min_dense_similarity,
            evidence_gate_mode=evidence_gate_mode,
            evidence_model=evidence_model,
            evidence_max_unsupported_claims=evidence_max_unsupported_claims,
            evidence_max_retries=evidence_max_retries,
            scope_gate_mode=scope_gate_mode,
            llm_provider=llm_provider,
            llm_model=required("LLM_MODEL"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            groq_api_key=groq_api_key,
            log_level=log_level,
        )


def configure_application_logging(level_name: str) -> None:
    """Ensure log_api diagnostics are visible under Uvicorn and Docker."""

    level = logging.getLevelNamesMapping().get(level_name.upper())
    if not isinstance(level, int):
        raise ValueError(f"invalid logging level: {level_name!r}")

    application_logger = logging.getLogger("log_api")
    application_logger.setLevel(level)
    if not application_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        application_logger.addHandler(handler)
    application_logger.propagate = False


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    retrieval_note: str | None = None


class RAGService:
    def __init__(self, config: ApiConfig):
        self.config = config
        self.retriever = HybridRetriever(
            RetrieverConfig(
                postgres_dsn=config.postgres_dsn,
                embedding_model=config.embedding_model,
                top_k=config.top_k,
                retrieval_mode=config.retrieval_mode,
                candidate_k=config.retrieval_candidate_k,
                trace_candidate_k=config.trace_candidate_k,
                reranker_model=config.reranker_model,
                rrf_k=config.rrf_k,
                dense_weight=config.dense_weight,
                lexical_weight=config.lexical_weight,
                trace_rerank_score_gap=config.trace_rerank_score_gap,
                trace_cosine_score_gap=config.trace_cosine_score_gap,
                temporal_top_k=config.temporal_top_k,
                max_trace_chunks=config.max_trace_chunks,
                confidence_gate_mode=config.confidence_gate_mode,
                min_dense_similarity=config.min_dense_similarity,
            )
        )
        self.qa_chain: QAChain | None = None
        self.evidence_verifier: EvidenceVerifier | None = None

    async def start(self) -> None:
        configure_application_logging(self.config.log_level)
        await self.retriever.start()

    async def close(self) -> None:
        await self.retriever.close()

    async def ask(self, question: str) -> AskResponse:
        scope = evaluate_scope_gate(
            question,
            mode=self.config.scope_gate_mode,
        )
        logger.info(
            "selfrag_scope_gate mode=%s eligible=%s would_abstain=%s "
            "category=%s reason=%s",
            scope.mode,
            scope.eligible,
            scope.would_abstain,
            scope.category,
            scope.reason,
        )
        if scope.enforced_abstention:
            return AskResponse(
                answer=deterministic_scope_answer(scope),
                retrieved_chunks=[],
                retrieval_note=scope.note,
            )

        retrieval = await self.retriever.retrieve_with_context(question)
        chunks = retrieval.chunks
        if not chunks:
            answer = deterministic_empty_context_answer(retrieval.note)
        else:
            if self.qa_chain is None:
                self.qa_chain = QAChain(
                    self.config.llm_provider,
                    self.config.llm_model,
                    self.config.ollama_base_url,
                    self.config.groq_base_url,
                    self.config.groq_api_key,
                    self.config.max_context_chars,
                )
            answer = await self.qa_chain.answer(question, chunks, retrieval.note)
            if self.config.evidence_gate_mode != "off":
                if self.evidence_verifier is None:
                    self.evidence_verifier = EvidenceVerifier(
                        self.config.llm_provider,
                        self.config.evidence_model or self.config.llm_model,
                        self.config.ollama_base_url,
                        self.config.groq_base_url,
                        self.config.groq_api_key,
                        self.config.max_context_chars,
                    )
                try:
                    verification = await self.evidence_verifier.verify(
                        question,
                        answer,
                        chunks,
                        retrieval.note,
                    )
                    requires_action = verification.requires_action(
                        self.config.evidence_max_unsupported_claims
                    )
                    retry_attempted = False
                    retry_succeeded = False
                    if (
                        self.config.evidence_gate_mode == "enforce"
                        and requires_action
                        and self.config.evidence_max_retries > 0
                    ):
                        retry_attempted = True
                        revised_answer = await self.qa_chain.revise(
                            question,
                            answer,
                            chunks,
                            retrieval.note,
                            verification.unsupported_claims,
                        )
                        revised_verification = await self.evidence_verifier.verify(
                            question,
                            revised_answer,
                            chunks,
                            retrieval.note,
                        )
                        if not revised_verification.requires_action(
                            self.config.evidence_max_unsupported_claims
                        ):
                            answer = revised_answer
                            verification = revised_verification
                            requires_action = False
                            retry_succeeded = True
                    logger.info(
                        "selfrag_evidence_gate mode=%s supported=%s "
                        "unsupported_claim_count=%d retry_attempted=%s "
                        "retry_succeeded=%s final_action=%s reason=%s",
                        self.config.evidence_gate_mode,
                        verification.supported,
                        verification.unsupported_claim_count,
                        retry_attempted,
                        retry_succeeded,
                        "abstain" if requires_action and self.config.evidence_gate_mode == "enforce" else "return_answer",
                        verification.reason,
                    )
                    if requires_action and self.config.evidence_gate_mode == "enforce":
                        answer = (
                            "The retrieved logs did not support all claims needed "
                            "for a reliable answer, so I cannot answer this safely "
                            "from the available evidence."
                        )
                except Exception:
                    # A verifier outage must not take down the QA endpoint. The
                    # failure is visible in logs and the original answer is kept.
                    logger.exception("selfrag evidence gate failed")
        return AskResponse(
            answer=answer,
            retrieved_chunks=chunks,
            retrieval_note=retrieval.note,
        )


def create_app(service: Any | None = None) -> FastAPI:
    """Create the API; an injected service makes endpoint tests LLM-free."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_service = service or RAGService(ApiConfig.from_env())
        if service is None:
            await active_service.start()
        app.state.rag_service = active_service
        try:
            yield
        finally:
            if service is None:
                await active_service.close()

    app = FastAPI(
        title="Sift Log QA API",
        description="Hybrid retrieval, reranking, and context-grounded question answering.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    async def ask(request: AskRequest) -> AskResponse:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be blank")
        try:
            return await app.state.rag_service.ask(question)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("QA request failed")
            raise HTTPException(status_code=502, detail="QA request failed") from exc

    return app


app = create_app()
