"""Gemini implementation of the LLMClient interface (google-genai SDK).

Receives exactly the same system prompt and user prompt (evidence passages + question) as the
Anthropic implementation; retrieval, prompting, citation checks and query rewriting are shared.
"""

from __future__ import annotations

from app.qa.llm import LLMError

# Finish reasons meaning the model declined or was stopped by a safety/policy filter.
_DECLINED = {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION"}


class GeminiLLM:
    # Gemini's thinking tokens count toward max_output_tokens, so the limit is higher than
    # the short answers need.
    def __init__(self, api_key: str, model: str, max_tokens: int = 8192) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for LLM_PROVIDER=gemini")
        try:
            import httpx
            from google import genai
            from google.genai import errors, types
        except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
            raise RuntimeError(
                "LLM_PROVIDER=gemini needs the optional dependency: pip install -e 'backend[gemini]'"
            ) from exc
        self.model = model
        self._max_tokens = max_tokens
        self._types, self._errors, self._httpx = types, errors, httpx
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))

    def complete(self, system: str, user: str) -> str:
        config = self._types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=self._max_tokens
        )
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=user, config=config
            )
        except self._errors.APIError as exc:
            raise LLMError(f"LLM API error {exc.code}: {exc.message}") from exc
        except self._httpx.TransportError as exc:
            raise LLMError(f"Could not reach the LLM API: {exc}") from exc

        feedback = response.prompt_feedback
        if feedback is not None and feedback.block_reason is not None:
            raise LLMError(f"The LLM declined to answer (prompt blocked: {feedback.block_reason.name})")
        finish = response.candidates[0].finish_reason if response.candidates else None
        finish_name = finish.name if finish is not None else None
        if finish_name in _DECLINED:
            raise LLMError(f"The LLM declined to answer (finish_reason={finish_name})")
        text = (response.text or "").strip()
        if not text:
            raise LLMError(f"The LLM returned no text (finish_reason={finish_name})")
        return text
