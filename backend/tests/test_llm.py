"""AnthropicLLM wrapper: response handling and error mapping, with a fake SDK client (no network)."""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.qa.llm import AnthropicLLM, LLMError

_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _response(stop_reason="end_turn", texts=("The deductible is Rs 10,000 [C1].",)):
    blocks = [SimpleNamespace(type="thinking", thinking="")]
    blocks += [SimpleNamespace(type="text", text=t) for t in texts]
    return SimpleNamespace(stop_reason=stop_reason, content=blocks)


class _FakeMessages:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _llm(model, result):
    llm = AnthropicLLM(api_key="test-key", model=model)
    plain, beta = _FakeMessages(result), _FakeMessages(result)
    llm._client = SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta))
    return llm, plain, beta


def test_returns_text_blocks_only():
    llm, _, _ = _llm("claude-opus-5", _response(texts=("Part one [C1].", " Part two [C2].")))
    assert llm.complete("sys", "user") == "Part one [C1]. Part two [C2]."


def test_sends_system_and_user_prompt_with_refusal_fallback():
    llm, plain, beta = _llm("claude-opus-5", _response())
    llm.complete("SYSTEM", "USER")
    assert plain.kwargs is None
    assert beta.kwargs["system"] == "SYSTEM"
    assert beta.kwargs["messages"] == [{"role": "user", "content": "USER"}]
    assert beta.kwargs["fallbacks"] == "default"
    assert beta.kwargs["betas"] == ["server-side-fallback-2026-07-01"]


def test_other_models_use_plain_endpoint():
    llm, plain, beta = _llm("claude-haiku-4-5", _response())
    llm.complete("s", "u")
    assert beta.kwargs is None
    assert plain.kwargs["model"] == "claude-haiku-4-5"
    assert "fallbacks" not in plain.kwargs


def test_refusal_raises():
    llm, _, _ = _llm("claude-opus-5", _response(stop_reason="refusal", texts=()))
    with pytest.raises(LLMError, match="declined"):
        llm.complete("s", "u")


def test_empty_reply_raises():
    llm, _, _ = _llm("claude-opus-5", _response(stop_reason="max_tokens", texts=("  ",)))
    with pytest.raises(LLMError, match="no text"):
        llm.complete("s", "u")


@pytest.mark.parametrize(
    "error, message",
    [
        (anthropic.APIConnectionError(request=_REQUEST), "Could not reach"),
        (
            anthropic.APIStatusError(
                "Overloaded", response=httpx2.Response(529, request=_REQUEST), body=None
            ),
            "529",
        ),
    ],
)
def test_sdk_errors_become_llm_errors(error, message):
    llm, _, _ = _llm("claude-opus-5", error)
    with pytest.raises(LLMError, match=message):
        llm.complete("s", "u")
