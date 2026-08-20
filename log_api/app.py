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

from log_api.qa import QAChain
from log_api.retriever import HybridRetriever, RetrievedChunk, RetrieverConfig


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
    reranker_model: str
    rrf_k: int
    dense_weight: float
    lexical_weight: float
    llm_provider: str
    llm_model: str
    ollama_base_url: str
    groq_base_url: str
    groq_api_key: str | None

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
        rrf_k = int(os.getenv("RRF_K", "60"))
        if rrf_k <= 0:
            raise RuntimeError("RRF_K must be positive")
        dense_weight = float(os.getenv("DENSE_RRF_WEIGHT", "1.0"))
        lexical_weight = float(os.getenv("LEXICAL_RRF_WEIGHT", "1.0"))
        if dense_weight < 0 or lexical_weight < 0:
            raise RuntimeError("RRF weights must be non-negative")
        if dense_weight == 0 and lexical_weight == 0:
            raise RuntimeError("at least one RRF weight must be positive")
        llm_provider = required("LLM_PROVIDER").lower()
        if llm_provider not in {"groq", "ollama"}:
            raise RuntimeError("LLM_PROVIDER must be either 'groq' or 'ollama'")

        groq_api_key = os.getenv("GROQ_API_KEY")
        if llm_provider == "groq" and not groq_api_key:
            raise RuntimeError("GROQ_API_KEY must be configured when LLM_PROVIDER='groq'")

        return cls(
            postgres_dsn=required("POSTGRES_DSN"),
            embedding_model=required("EMBEDDING_MODEL"),
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            retrieval_candidate_k=retrieval_candidate_k,
            reranker_model=os.getenv(
                "RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
            llm_provider=llm_provider,
            llm_model=required("LLM_MODEL"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            groq_api_key=groq_api_key,
        )


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunk]


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
                reranker_model=config.reranker_model,
                rrf_k=config.rrf_k,
                dense_weight=config.dense_weight,
                lexical_weight=config.lexical_weight,
            )
        )
        self.qa_chain: QAChain | None = None

    async def start(self) -> None:
        await self.retriever.start()

    async def close(self) -> None:
        await self.retriever.close()

    async def ask(self, question: str) -> AskResponse:
        chunks = await self.retriever.retrieve(question)
        if self.qa_chain is None:
            self.qa_chain = QAChain(
                self.config.llm_provider,
                self.config.llm_model,
                self.config.ollama_base_url,
                self.config.groq_base_url,
                self.config.groq_api_key,
            )
        answer = await self.qa_chain.answer(question, chunks)
        return AskResponse(answer=answer, retrieved_chunks=chunks)


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
