"""LLM client interface and the Anthropic implementation."""

from __future__ import annotations

from typing import Protocol

import anthropic


class LLMError(RuntimeError):
    """The LLM call failed or was declined; callers fall back to evidence-only output."""


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str:
        """Return the model's text reply or raise LLMError."""


# Models that support the server-side refusal fallback parameter.
_FALLBACK_MODELS = {"claude-opus-5", "claude-opus-5-5", "claude-fable-5-1"}


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, max_tokens: int = 2048) -> None:
        self.model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key, timeout=60.0)

    def complete(self, system: str, user: str) -> str:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        try:
            if self.model in _FALLBACK_MODELS:
                response = self._client.beta.messages.create(
                    **kwargs,
                    output_config={"effort": "low"},
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            else:
                response = self._client.messages.create(**kwargs)
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the LLM API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"LLM API error {exc.status_code}: {exc.message}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("The LLM declined to answer")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise LLMError(f"The LLM returned no text (stop_reason={response.stop_reason})")
        return text
