"""Claude-only path: no deterministic templates, all sources reach the prompt, no network."""

from rag import claude_chat
from rag.schemas import AccidentScenario, ChatRequest, SourceDoc

SOURCES = [
    SourceDoc(content="급박한 위험이 있을 때에는 즉시 작업을 중지시키고", metadata={"law_name": "산업안전보건법", "article": "제51조", "page": 0}),
    SourceDoc(content="비상구를 설치하여야 한다", metadata={"law_name": "산업안전보건기준에 관한 규칙", "article": "제17조", "page": 3}),
]


def test_claude_path_answers_from_every_source(monkeypatch) -> None:
    sent = {}
    monkeypatch.setattr(claude_chat, "retrieve_integrated", lambda *args, **kwargs: SOURCES)

    def fake_call(messages):
        sent["messages"] = messages
        return "결론: 위반에 해당합니다.\n근거: 산업안전보건법 제51조"

    monkeypatch.setattr(claude_chat, "call_anthropic_blocking", fake_call)

    response = claude_chat.claude_rag_chat(
        ChatRequest(
            question="리튬전지 화재에서 작업중지 위반인가?",
            scenario=AccidentScenario(overview="리튬전지 화재", details="전조 5차례", workers="23명 사망"),
            mode="scenario",
        )
    )

    system, user = sent["messages"]
    assert system["role"] == "system" and "결론: " in system["content"]
    assert "[사고 시나리오]" in user["content"] and "전조 5차례" in user["content"]
    assert "산업안전보건법 제51조 (p.0)" in user["content"]
    assert "산업안전보건기준에 관한 규칙 제17조 (p.3)" in user["content"]
    assert response.answer.startswith("결론: ")
    assert response.sources == SOURCES
    assert response.graph_trace["execution"] == "claude_rag_generation"
    assert response.citation_check["status"] == "pass"
