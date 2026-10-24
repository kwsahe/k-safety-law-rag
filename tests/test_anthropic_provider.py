"""Claude provider adapter: request shape and refusal handling, no network."""

from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")

from rag import chatbot  # noqa: E402


class FakeClient:
    def __init__(self, response):
        self.kwargs = None
        self.response = response
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self.response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _response(stop_reason: str, *texts: str):
    blocks = [SimpleNamespace(type="thinking", thinking="")]
    blocks += [SimpleNamespace(type="text", text=text) for text in texts]
    return SimpleNamespace(stop_reason=stop_reason, stop_details=None, content=blocks)


def test_system_prompt_moves_to_top_level_and_qwen_switch_is_dropped(monkeypatch) -> None:
    client = FakeClient(_response("end_turn", "답변", " 끝"))
    monkeypatch.setattr(anthropic, "Anthropic", lambda: client)

    answer = chatbot.call_anthropic_blocking(
        [
            {"role": "system", "content": "시스템"},
            {"role": "user", "content": "/no_think\n\n질문"},
        ]
    )

    assert answer == "답변 끝"
    assert client.kwargs["system"] == "시스템"
    assert client.kwargs["messages"] == [{"role": "user", "content": "질문"}]
    assert "temperature" not in client.kwargs
    assert client.kwargs["fallbacks"] == "default"


def test_refusal_raises_instead_of_returning_partial_text(monkeypatch) -> None:
    monkeypatch.setattr(anthropic, "Anthropic", lambda: FakeClient(_response("refusal", "부분")))

    with pytest.raises(RuntimeError, match="declined"):
        chatbot.call_anthropic_blocking([{"role": "user", "content": "질문"}])
