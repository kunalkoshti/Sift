"""Environment-driven configuration for the independent evaluation LLM."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class EvaluationLLMConfig:
    """Connection and model settings for classifier/RAGAS calls.

    The evaluation model is intentionally separate from the model used by
    ``log-api``.  Any OpenAI-compatible provider can be selected by changing
    the EVAL_* variables in ``.env``.
    """

    provider: str
    model: str
    base_url: str
    api_key: str | None
    reasoning_effort: str | None = None

    @classmethod
    def from_env(cls) -> "EvaluationLLMConfig":
        provider = os.getenv("EVAL_PROVIDER") or os.getenv("LLM_PROVIDER", "groq")
        provider = provider.lower()
        model = os.getenv("EVAL_MODEL") or os.getenv(
            "LLM_MODEL", "openai/gpt-oss-20b"
        )

        if provider == "ollama":
            base_url = os.getenv("EVAL_BASE_URL") or os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).rstrip("/") + "/v1"
            api_key = "ollama"
        else:
            base_url = os.getenv("EVAL_BASE_URL") or os.getenv(
                "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
            )
            api_key = os.getenv("EVAL_API_KEY") or os.getenv("GROQ_API_KEY")

        configured_reasoning_effort = os.getenv("EVAL_REASONING_EFFORT")
        if configured_reasoning_effort:
            reasoning_effort = configured_reasoning_effort
        elif provider == "groq":
            # Preserve the existing Groq GPT-OSS behavior for legacy .env files.
            reasoning_effort = "low"
        else:
            # Cerebras and other providers may reject this GPT-OSS-specific
            # parameter, so it is omitted unless explicitly configured.
            reasoning_effort = None

        if provider not in {"groq", "cerebras", "openai", "openai-compatible", "ollama"}:
            raise ValueError(
                "unsupported EVAL_PROVIDER={!r}; choose groq, cerebras, "
                "openai, openai-compatible, or ollama".format(provider)
            )

        if provider != "ollama" and not api_key:
            raise RuntimeError(
                "EVAL_API_KEY is required for evaluation provider {!r}".format(provider)
            )

        return cls(
            provider=provider,
            model=model,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )

    def build_client(self):
        """Build the OpenAI-compatible async client used by eval components."""

        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def request_options(self) -> dict[str, str]:
        """Return provider-specific optional completion parameters."""

        if self.reasoning_effort is None:
            return {}
        return {"reasoning_effort": self.reasoning_effort}
