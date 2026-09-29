"""Grounded Q&A and LLM providers."""

from __future__ import annotations

import logging

from app.config import Settings
from app.qa.llm import LLMClient

logger = logging.getLogger(__name__)


def build_llm(settings: Settings) -> LLMClient | None:
    """The configured LLM provider, or None (extractive mode) when its API key is not set."""
    provider = settings.llm_provider
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            return None
        from app.qa.llm import AnthropicLLM

        return AnthropicLLM(settings.anthropic_api_key, settings.llm_model)
    if provider == "gemini":
        if not settings.gemini_api_key:
            logger.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set; answers are evidence-only")
            return None
        from app.qa.gemini_llm import GeminiLLM

        return GeminiLLM(settings.gemini_api_key, settings.gemini_model)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (use 'anthropic' or 'gemini')")
