"""LLM client for the RAG pipeline.

We use a Protocol so the pipeline can be tested without hitting Groq, and so
the LLM can be swapped (e.g., for OpenRouter or a local model) without
touching the rest of the codebase.
"""
from __future__ import annotations

import time
from typing import Optional, Protocol

from config import settings


class LLMClient(Protocol):
    """Interface for any LLM that the pipeline can call."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Synchronous: take prompts, return the model's text response."""
        ...


class GroqLLMClient:
    """Groq client using the official `groq` Python SDK.

    Groq's free tier serves Llama, Mixtral, and Gemma models at very high
    throughput, which keeps p95 latency low without compromising quality.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
    ):
        # Resolve config lazily so tests can run without a real key
        self._api_key = api_key or settings.groq_api_key
        self._model = model or settings.llm_model
        self._max_retries = max_retries
        self._timeout = timeout_seconds
        self._client = None  # Lazy init

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. "
                    "Get a free key from https://console.groq.com and add it to .env"
                )
            from groq import Groq
            self._client = Groq(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        client = self._ensure_client()
        max_tokens = max_tokens if max_tokens is not None else settings.max_answer_tokens
        temperature = temperature if temperature is not None else settings.llm_temperature

        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content
                return (content or "").strip()
            except Exception as e:
                last_exc = e
                # Exponential backoff on transient errors (rate limits, network blips).
                # Last attempt re-raises.
                if attempt < self._max_retries:
                    time.sleep(0.5 * (2 ** (attempt - 1)))

        raise RuntimeError(f"Groq generation failed after {self._max_retries} attempts: {last_exc}")
