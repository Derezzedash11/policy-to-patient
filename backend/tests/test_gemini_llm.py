"""Gemini provider: configuration, provider selection and SDK invocation (fake client, no network).

Responses and errors are real google-genai types; only `client.models.generate_content` is faked.
"""

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types

from app.config import Settings
from app.qa import build_llm
from app.qa.answer import answer_question
from app.qa.gemini_llm import GeminiLLM
from app.qa.llm import AnthropicLLM, LLMError
from app.qa.prompt import SYSTEM_PROMPT


def _response(text="The deductible is Rs 10,000 [C1].", finish="STOP", thought=None, block=None):
    parts = ([{"text": thought, "thought": True}] if thought else []) + ([{"text": text}] if text else [])
    data: dict = {"candidates": [{"content": {"role": "model", "parts": parts}, "finish_reason": finish}]}
    if block:
        data = {"candidates": [], "prompt_feedback": {"block_reason": block}}
    return types.GenerateContentResponse.model_validate(data)


class _FakeModels:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _gemini(result, model="gemini-test-model") -> tuple[GeminiLLM, _FakeModels]:
    llm = GeminiLLM(api_key="test-key", model=model)
    fake = _FakeModels(result)
    llm._client = SimpleNamespace(models=fake)
    return llm, fake


# --------------------------------------------------------------------------- configuration


def test_settings_defaults_keep_anthropic(monkeypatch):
    for var in ("LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.llm_provider == "anthropic"
    assert s.gemini_api_key is None
    assert s.gemini_model == "gemini-2.5-flash"


def test_settings_read_gemini_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", " Gemini ")
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom")
    s = Settings()
    assert (s.llm_provider, s.gemini_api_key, s.gemini_model) == ("gemini", "env-key", "gemini-custom")


# --------------------------------------------------------------------------- provider selection


def test_selects_gemini_with_configured_model(tmp_path):
    llm = build_llm(Settings(data_dir=tmp_path, llm_provider="gemini", gemini_api_key="k",
                             gemini_model="gemini-custom"))
    assert isinstance(llm, GeminiLLM) and llm.model == "gemini-custom"


def test_selects_anthropic_by_default(tmp_path):
    llm = build_llm(Settings(data_dir=tmp_path, llm_provider="anthropic", anthropic_api_key="k",
                             llm_model="claude-opus-5"))
    assert isinstance(llm, AnthropicLLM) and llm.model == "claude-opus-5"


def test_gemini_key_is_not_used_by_anthropic_provider_and_vice_versa(tmp_path):
    assert build_llm(Settings(data_dir=tmp_path, llm_provider="anthropic", anthropic_api_key=None,
                              gemini_api_key="k")) is None
    assert build_llm(Settings(data_dir=tmp_path, llm_provider="gemini", gemini_api_key=None,
                              anthropic_api_key="k")) is None


def test_unknown_provider_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        build_llm(Settings(data_dir=tmp_path, llm_provider="nope"))


# --------------------------------------------------------------------------- missing GEMINI_API_KEY


def test_missing_gemini_key_means_extractive_mode(tmp_path, caplog):
    assert build_llm(Settings(data_dir=tmp_path, llm_provider="gemini", gemini_api_key=None)) is None
    assert "GEMINI_API_KEY is not set" in caplog.text


def test_missing_gemini_key_reported_by_health(tmp_path):
    from fastapi.testclient import TestClient

    from app.embeddings.hashing import HashingEmbedder
    from app.main import create_app

    settings = Settings(data_dir=tmp_path, llm_provider="gemini", gemini_api_key=None)
    health = TestClient(create_app(settings, embedder=HashingEmbedder())).get("/health").json()
    assert health["llm_mode"] == "extractive" and health["llm_model"] is None


def test_gemini_client_refuses_empty_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiLLM(api_key="", model="gemini-test-model")


def test_key_is_not_exposed_in_repr_or_errors():
    llm, _ = _gemini(errors.APIError(401, {"error": {"code": 401, "message": "API key not valid"}}))
    with pytest.raises(LLMError) as exc:
        llm.complete("s", "u")
    assert "test-key" not in str(exc.value) and "test-key" not in repr(llm)


# --------------------------------------------------------------------------- invocation


def test_sends_system_prompt_user_prompt_and_model():
    llm, fake = _gemini(_response())
    assert llm.complete("SYSTEM", "USER") == "The deductible is Rs 10,000 [C1]."
    assert fake.kwargs["model"] == "gemini-test-model"
    assert fake.kwargs["contents"] == "USER"
    assert fake.kwargs["config"].system_instruction == "SYSTEM"
    assert fake.kwargs["config"].max_output_tokens == 8192


def test_thought_parts_are_not_returned():
    llm, _ = _gemini(_response(text="Answer [C1].", thought="internal reasoning"))
    assert llm.complete("s", "u") == "Answer [C1]."


@pytest.mark.parametrize("finish", ["SAFETY", "PROHIBITED_CONTENT", "RECITATION"])
def test_safety_stops_raise(finish):
    llm, _ = _gemini(_response(text="partial", finish=finish))
    with pytest.raises(LLMError, match="declined"):
        llm.complete("s", "u")


def test_blocked_prompt_raises():
    llm, _ = _gemini(_response(block="SAFETY"))
    with pytest.raises(LLMError, match="prompt blocked: SAFETY"):
        llm.complete("s", "u")


def test_empty_reply_raises():
    llm, _ = _gemini(_response(text=None, thought="only thinking", finish="MAX_TOKENS"))
    with pytest.raises(LLMError, match="no text.*MAX_TOKENS"):
        llm.complete("s", "u")


@pytest.mark.parametrize(
    "error, message",
    [
        (errors.ClientError(429, {"error": {"code": 429, "message": "Resource exhausted"}}), "429"),
        (errors.ServerError(503, {"error": {"code": 503, "message": "Unavailable"}}), "503"),
        (httpx.ConnectError("connection refused"), "Could not reach"),
    ],
)
def test_sdk_errors_become_llm_errors(error, message):
    llm, _ = _gemini(error)
    with pytest.raises(LLMError, match=message):
        llm.complete("s", "u")


# --------------------------------------------------------------------------- shared Q&A pipeline


def test_gemini_gets_the_same_grounded_prompt_and_verification(indexed_retriever):
    llm, fake = _gemini(_response("A deductible of Rs 10,000 applies to each policy year [C1]."))
    res = answer_question("What is the deductible each policy year?", "testdoc", indexed_retriever,
                          llm, top_k=3, min_score=0.10)
    assert res.status == "answered" and res.mode == "generative"
    assert res.citations[0].page == 4
    assert fake.kwargs["config"].system_instruction == SYSTEM_PROMPT
    assert "[C1] page 4, section: 3. DEDUCTIBLE AND CO-PAYMENT" in fake.kwargs["contents"]


def test_gemini_answers_go_through_the_same_figure_check(indexed_retriever):
    llm, _ = _gemini(_response("The deductible is Rs 25,000 [C1]."))
    res = answer_question("What is the deductible each policy year?", "testdoc", indexed_retriever,
                          llm, top_k=3, min_score=0.10)
    assert res.status == "manual_review" and res.answer is None


def test_gemini_errors_fall_back_to_evidence(indexed_retriever):
    llm, _ = _gemini(errors.ServerError(503, {"error": {"code": 503, "message": "Unavailable"}}))
    res = answer_question("What is the deductible each policy year?", "testdoc", indexed_retriever,
                          llm, top_k=3, min_score=0.10)
    assert res.status == "llm_error" and res.citations
