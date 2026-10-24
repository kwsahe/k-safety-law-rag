"""Claude-only answer path (LLM_PROVIDER=anthropic).

Reuses the existing integrated retrieval, but skips the deterministic answer
templates written for EXAONE/Qwen and passes every retrieved source to Claude
with its own prompt. The EXAONE/Qwen path in rag.chatbot is left unchanged.
"""

from __future__ import annotations

from rag.chatbot import (
    _resolve_scenario,
    build_retrieval_query,
    call_anthropic_blocking,
    format_scenario,
    retrieve_integrated,
)
from rag.citation_validator import validate_answer_citations
from rag.config import RAG_TOP_K
from rag.question_graph import public_graph_trace, run_question_graph
from rag.scenario_analysis import format_scenario_profile
from rag.schemas import AccidentScenario, ChatRequest, ChatResponse, SourceDoc

CLAUDE_SYSTEM_PROMPT = """당신은 한국 산업안전보건법령과 중대재해처벌법을 다루는 법령 상담 보조자입니다.
사용자는 사고를 조사하거나 재발을 막으려는 사업장 관리자와 안전 담당자입니다. 아래 원칙으로 답하세요.

[근거]
- 조항을 인용할 때는 사용자 메시지의 [검색된 법령 근거]에 실제로 있는 조항만 쓰세요. 기억에 의존해 조항 번호, 형량, 금액을 만들지 마세요.
- 판단에 필요한 조항이 검색 근거에 없으면 검색된 근거에 없어 추가 확인이 필요하다고 밝히고, 조항 번호 없이 판단 방향만 설명하세요.
- 조항은 항상 법령 전체 이름과 함께 쓰세요. 예: 산업안전보건법 제51조, 산업안전보건기준에 관한 규칙 제17조, 중대재해처벌법 시행령 제4조제3호. "같은 법", "동법", "시행령 제4조"처럼 줄이지 마세요.
- 검색 근거에는 질문과 무관한 조문(다른 업종이나 다른 작업에 관한 규정 등)이 섞여 있을 수 있습니다. 사고 사실과 맞는 것만 쓰세요.
- 사고 사실은 [사고 시나리오]와 [구조화된 사고 사실]에 있는 것만 사용하세요. 없는 사실은 추정하지 말고 확인할 사항으로 남기세요.

[판단]
- 위반·적용·책임 여부를 물으면 결론을 먼저 분명히 밝히세요: 해당한다, 해당하지 않는다, 사실 확인이 더 필요하다 중 하나.
- 요건을 하나씩 사고 사실에 대입해 판단하세요. 수치 기준이 있으면 "굴착 깊이 4m ≥ 기준 2m"처럼 비교를 보여주세요.
- 서류상 형식과 실제 이행을 구분하고, 계약 명칭보다 실질적인 지휘·지배·관리 관계로 책임 주체를 판단하세요.
- 법 해석이 갈리거나 법원 판단이 달라질 수 있는 지점은 단정하지 말고 쟁점으로 제시하세요.
- 질문에 여러 항목이 있으면 모두 빠짐없이 답하세요.

[화면 형식]
답변은 줄 단위로 그대로 표시되는 화면에 나갑니다.
- 첫 줄은 "결론: "으로 시작하는 한두 문장입니다.
- 섹션 제목은 "[제목]" 한 줄 또는 "1. 제목" 형식으로 쓰고, 목록은 "- ", 강조는 **굵게**만 사용하세요.
- #으로 시작하는 마크다운 제목, 표(|), 코드 블록은 쓰지 마세요. 화면에 기호가 그대로 보입니다.
- "[근거 3]" 같은 내부 번호 대신 법령명·조항·페이지로 출처를 적으세요. 예: (산업안전보건법 제51조, p.0)
- 마지막 줄은 "근거: "로 시작해 인용한 조항을 모아 적으세요."""


def format_claude_sources(sources: list[SourceDoc]) -> str:
    blocks = []
    for index, source in enumerate(sources, 1):
        metadata = source.metadata
        ref = " ".join(
            str(value) for value in (metadata.get("law_name"), metadata.get("article") or metadata.get("annex")) if value
        )
        blocks.append(f"[근거 {index}] {ref} (p.{metadata.get('page', '?')})\n{source.content.strip()}")
    return "\n\n".join(blocks)


def build_claude_messages(
    question: str,
    sources: list[SourceDoc],
    *,
    scenario: AccidentScenario | None = None,
    scenario_profile: dict | None = None,
) -> list[dict[str, str]]:
    parts = []
    scenario_text = format_scenario(scenario) if scenario else ""
    if scenario_text:
        parts.append(f"[사고 시나리오]\n{scenario_text}")
    profile_text = format_scenario_profile(scenario_profile)
    if profile_text:
        parts.append(f"[구조화된 사고 사실]\n{profile_text}")
    parts.append(f"[검색된 법령 근거]\n{format_claude_sources(sources)}")
    parts.append(f"[질문]\n{question}")
    return [
        {"role": "system", "content": CLAUDE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def claude_rag_chat(request: ChatRequest) -> ChatResponse:
    """Retrieve as rag_chat does, then let Claude answer from every retrieved source."""
    scenario = _resolve_scenario(request.scenario)
    retrieval_query = build_retrieval_query(request.question, scenario, request.scenario_profile)
    graph_state = run_question_graph(
        request.question,
        mode=request.mode,
        cache_context=retrieval_query,
        scenario_profile=request.scenario_profile,
    )
    sources = retrieve_integrated(retrieval_query, text_top_k=RAG_TOP_K, table_top_k=RAG_TOP_K, table_first=True)
    messages = build_claude_messages(
        request.question,
        sources,
        scenario=scenario,
        scenario_profile=request.scenario_profile,
    )
    answer = call_anthropic_blocking(messages).strip()
    # The graph still records scope/intent, but this path never runs its deterministic route.
    trace = {**public_graph_trace(graph_state), "route": "claude_rag", "execution": "claude_rag_generation"}
    return ChatResponse(
        answer=answer,
        sources=sources,
        citation_check=validate_answer_citations(answer, sources),
        graph_trace=trace,
    )
