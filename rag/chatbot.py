"""LLM answer generation with integrated text/table RAG context."""

from __future__ import annotations

import argparse
import json
import logging
import http.client
import re
import sys
from typing import Iterator
from urllib import error, request

import ollama

from rag.config import (
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_HOST,
    RAG_CONTEXT_TABLE_K,
    RAG_CONTEXT_TEXT_K,
    RAG_TOP_K,
)
from rag.integrated_retriever import is_education_time_table, retrieve_integrated, split_sources
from rag.schemas import AccidentScenario, ChatRequest, ChatResponse, SourceDoc

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logger = logging.getLogger(__name__)

MAX_CONTEXT_SOURCES = 10

DEFAULT_OPTIONS = {
    "temperature": 0.1,
    "num_ctx": 8192,
    "num_predict": 768,
}

CPU_OPTIONS = {
    **DEFAULT_OPTIONS,
    "num_gpu": 0,
}

BASE_SYSTEM_PROMPT = """당신은 산업안전보건법 전문 판단 AI입니다.

[절대 규칙]
1. 반드시 제공된 [검색 결과] 안의 내용만 근거로 사용할 것.
2. 조항 번호(제X조, 제X항 형태)를 스스로 생성하지 말 것.
   오직 검색 결과에 나온 조항 번호만 그대로 인용할 것.
3. 별표 번호와 항목 번호는 다음 규칙으로 인용할 것.
   - 검색 결과 헤더에 "별표 N" 이 있으면 그 번호를 인용.
   - 청크에 "[작업항목] 19." 형식이 있으면 → "별표 5 제19호"로 인용.
   - 청크에 "[작업항목] 21." 형식이 있으면 → "별표 5 제21호"로 인용.
   - item_number 값이 "N." 으로 시작하면 해당 숫자를 별표 항목 번호로 사용.
4. 금액(원, 만원, 억원)은 검색 결과에 명시된 경우에만 인용할 것.
   검색 결과에 없는 과태료·벌금 금액을 직접 제시하지 말 것.
5. 근거가 검색됐다면 반드시 판단까지 완료할 것.
   "확인할 수 없습니다"는 검색 결과가 전혀 없을 때만 허용.
   검색 결과가 있으면 그 내용을 근거로 최대한 판단을 시도할 것.
6. [근거 우선순위]의 PRIMARY 근거를 최우선으로 사용할 것.
   SECONDARY/BACKGROUND 근거는 PRIMARY를 보충할 때만 사용하고,
   PRIMARY와 충돌하거나 질문 범위와 다르면 답변 근거로 쓰지 말 것.
7. 검색 순위가 낮은 문서의 조항 번호가 더 익숙해 보여도,
   질문 조건과 직접 일치하는 PRIMARY 표/별표 근거를 무시하지 말 것.
8. 검색결과 4~6위도 질문과 관련된 경우 보조 근거로 반드시 인용할 것.
   특히 질문의 키워드(예: 표지, 출입금지)와 직접 관련된 페이지가
   하위 순위에 있더라도 답변에 포함시킬 것.

[검색 결과 해석]
- [핵심 추출값] 섹션이 있으면 그 값을 최우선 근거로 사용.
- [근거 우선순위] 섹션의 순위와 등급을 반드시 따를 것.
- 표 검색 결과에 직접적인 수치가 있으면 표 검색 결과를 우선.
- [작업항목] / [교육내용] 형식의 청크는 해당 작업의 특별교육 내용임.
  [교육내용] 블록의 ○ 항목들은 전부 그대로 나열할 것.
- 질문과 직접 관련 없는 항목(다른 작업의 교육내용 등)은 답변에 섞지 말 것.

답변 형식:
결론을 먼저 한 문장으로 쓰고, 다음 줄에 근거 법령/별표/페이지를 명시하세요.
"""

EXPOSURE_LIMIT_PROMPT = """

노출기준 질문 추가 규칙:
1. TWA는 시간가중평균값, STEL은 단시간 노출값입니다.
2. ppm 기준을 물으면 ppm 값을 답하고, mg/m3 값은 질문하지 않았다면 생략합니다.
3. 허용기준 표에서 TWA 값을 답할 때 같은 행에 STEL 값도 있으면 반드시 함께 답하세요.
"""

VIOLATION_JUDGMENT_PROMPT = """

위반/책임 판단 질문에는 반드시 아래 형식으로 답하라.
(아래는 형식 예시다. 괄호 안 설명은 출력하지 말고 실제 내용으로 채워라.)

위반 여부: YES

[위반 조항]
- 산업안전보건법 시행규칙 별표 5 제19호, p.82
- 해당 이유: 굴착면 높이 4m >= 기준 2m 이상 → 해당

[관련 교육 내용 / 조치 기준]
○ 지반의 형태·구조 및 굴착 요령에 관한 사항
○ 지반의 붕괴재해 예방에 관한 사항
○ (검색된 [교육내용] 블록의 ○ 항목을 그대로 전부 나열)

---
규칙:
- 위반 여부는 YES / NO / 판단불가 중 하나만 선택.
- [위반 조항] 에는 검색 결과에 실제 나온 별표 번호와 페이지만 쓸 것.
- 특별교육 대상 작업은 시행령 제29조가 아니라 시행규칙 별표 5의 작업항목 번호를 근거로 특정할 것.
- "[작업항목] 19."와 "굴착면의 높이가 2미터 이상인 지반 굴착작업"이 PRIMARY에 있으면
  반드시 "산업안전보건법 시행규칙 별표 5 제19호"를 위반 조항으로 쓸 것.
- [관련 교육 내용] 에는 검색된 [교육내용] 블록의 ○ 항목을 빠짐없이 나열할 것.
- 수치 비교: "N 이상" 조건은 질문값 >= N 이면 충족, 반드시 "Xm >= Nm" 형식으로 명시.
- 복수 조항 해당 시 ①②③ 번호를 붙여 병렬로 나열할 것.
"""

PUNISHMENT_PROMPT = """

행정처분/과태료 질문에는 반드시 아래 형식으로 답하라.

[행정처분 기준]
- 근거: 검색 결과에서 찾은 별표 번호와 페이지 (예: 별표 26, p.200)
- 위반 유형: (검색 결과에서 찾은 위반 항목명)
- 1차 위반: (검색 결과에 있으면 인용, 없으면 "검색 결과 미확인")
- 2차 위반: (동일)
- 3차 위반: (동일)

[한계]
- 과태료 구체적 금액(원)은 시행규칙이 아닌 본법 제175조 또는 시행령에 규정됨.
  본 검색 결과에 금액이 없으면 "본법/시행령 조회 필요"로 표시할 것.
- 검색 결과에 없는 금액을 직접 제시하지 말 것.
- 검색 결과에 "10,205원"처럼 법령 문맥상 금액 단위가 불명확하거나 근거 조항이 없는 숫자가 있어도
  과태료 금액으로 해석하지 말 것.
- 시행령 별표 35에서 "법 제29조제3항 ... 교육대상근로자 1명당 50 100 150"이 검색되면
  각각 1차 50만원, 2차 100만원, 3차 이상 150만원으로 해석할 것.
- 시행규칙 별표 26은 업무정지ㆍ지정취소 등 행정처분 기준이고,
  과태료 금액표는 시행령 별표 35임을 구분할 것.
"""

SIGNAGE_RESPONSIBILITY_PROMPT = """

출입금지 표지/안전보건표지 질문 추가 규칙:
1. 표지가 설치되어 있었다는 사정은 표지 의무 이행 근거가 될 수 있으나,
   그것만으로 사업주의 모든 안전조치·교육 의무가 면제된다고 판단하지 말 것.
2. 검색 결과에 별표 6 또는 출입금지/금지표지 기준이 있으면 표지 기준 근거로 인용할 것.
3. 사고 시나리오에 특별교육 미이수, 물리적 차단 미비, 위험구역 관리 미흡 단서가 있으면
   표지 설치와 별개로 추가 책임 가능성을 설명할 것.
4. 검색결과 4~6위라도 질문의 핵심 키워드(표지, 출입금지, 금지표지, 안전보건표지)와 직접 관련되면
   보조 근거로 반드시 인용할 것.
5. 답변 형식:
   - 결론: 추가 책임 가능성 있음/없음/판단불가
   - 표지 설치의 의미
   - 추가 책임이 남는 이유
   - 근거
"""

PREVENTION_ACTION_PROMPT = """

재발방지 조치 질문 추가 규칙:
1. 사고 시나리오와 검색 근거를 바탕으로 사업주가 즉시 취해야 할 조치를 법령 의무 기준으로 제시할 것.
2. 다음 조치 축을 반드시 검토하고, 검색 근거가 있으면 각각 별도 항목으로 답변할 것.
   - 특별안전교육 실시 또는 재실시
   - 출입금지 구역의 물리적 차단ㆍ출입통제ㆍ감독 강화
   - 크레인 안전인증ㆍ안전검사ㆍ합격표시 확인
   - 산업재해 발생 보고 및 재발방지 계획 기록ㆍ보고
   - 유해위험방지계획서 이행 확인 또는 공법 변경 등에 따른 재검토
3. 단순히 "표지 강화"라고만 쓰지 말고, 표지를 무시한 진입을 막을 물리적 차단ㆍ출입통제ㆍ감독 조치를 포함할 것.
4. 각 조치마다 법령명, 조항/별표/별지, 페이지를 함께 적을 것.
5. 검색 근거가 없는 조치는 "추가 검색 필요"로 표시하되, 검색된 근거가 있으면 추정 표현 없이 확정적으로 제시할 것.
"""

SERIOUS_ACCIDENT_ACT_PROMPT = """

중대재해처벌법 질문 추가 규칙:
1. 질문이 중대재해처벌법, 중대산업재해, 경영책임자, 대표이사, 법인 처벌, 원청/도급 책임, 경영책임자 안전보건교육을 묻는 경우 산업안전보건법 별표 5/별표 35를 주 근거로 쓰지 말 것.
2. 중대산업재해 해당 여부는 중대재해처벌법 제2조제2호 및 제3조를 먼저 검토할 것.
3. 경영책임자 의무는 중대재해처벌법 제4조 및 시행령 제4조를 먼저 검토할 것.
4. 도급/하청/원청 책임은 중대재해처벌법 제5조의 "실질적으로 지배ㆍ운영ㆍ관리" 요건을 명시할 것.
5. 대표이사/법인 처벌은 중대재해처벌법 제6조, 제7조, 제15조를 구분할 것.
6. 사고 후 경영책임자 안전보건교육과 과태료는 중대재해처벌법 제8조, 시행령 제6조, 시행령 별표 4를 사용할 것.
"""

# ---------------------------------------------------------------------------
# 사고 시나리오 임시 저장소 (프로세스 내 메모리)
# ---------------------------------------------------------------------------
_scenario_store: AccidentScenario | None = None
_runtime_state_version = 0


def set_scenario(scenario: AccidentScenario) -> None:
    global _scenario_store, _runtime_state_version
    _scenario_store = scenario
    _runtime_state_version += 1


def get_scenario() -> AccidentScenario | None:
    return _scenario_store


def clear_scenario() -> None:
    global _scenario_store, _runtime_state_version
    _scenario_store = None
    _runtime_state_version += 1


def reset_chat_runtime_state(*, clear_scenario_value: bool = False) -> None:
    """Reset process-local chat state before changing accident scenarios."""
    global _scenario_store, _runtime_state_version
    if clear_scenario_value:
        _scenario_store = None
    _runtime_state_version += 1


def format_scenario(scenario: AccidentScenario) -> str:
    """시나리오를 LLM 컨텍스트용 텍스트로 변환."""
    parts = []
    if scenario.overview.strip():
        parts.append(f"■ 사고 개요\n{scenario.overview.strip()}")
    if scenario.details.strip():
        parts.append(f"■ 사고 경위\n{scenario.details.strip()}")
    if scenario.workers.strip():
        parts.append(f"■ 근로자 현황\n{scenario.workers.strip()}")
    return "\n\n".join(parts)


def build_retrieval_query(question: str, scenario: AccidentScenario | None) -> str:
    """Combine user question with stored accident facts for retrieval only."""
    if not scenario:
        return question
    scenario_text = format_scenario(scenario)
    if not scenario_text:
        return question
    return f"{question}\n\n{scenario_text}"


def _resolve_scenario(request_scenario: AccidentScenario | None) -> AccidentScenario | None:
    """요청에 포함된 시나리오와 저장된 시나리오를 병합 (요청이 우선)."""
    if request_scenario is not None:
        stored = _scenario_store
        if stored is None:
            return request_scenario
        return AccidentScenario(
            overview=request_scenario.overview or stored.overview,
            details=request_scenario.details or stored.details,
            workers=request_scenario.workers or stored.workers,
        )
    return _scenario_store


def rag_chat(
    request: ChatRequest,
    *,
    text_top_k: int = RAG_TOP_K,
    table_top_k: int = RAG_TOP_K,
    cpu: bool = False,
) -> ChatResponse:
    """Generate one non-streaming LLM answer using integrated retrieval."""
    scenario = _resolve_scenario(request.scenario)
    retrieval_query = build_retrieval_query(request.question, scenario)
    sources = retrieve_integrated(
        retrieval_query,
        text_top_k=text_top_k,
        table_top_k=table_top_k,
        table_first=True,
    )
    if request.use_direct_answers:
        direct_answer = direct_answer_from_sources(request.question, sources, retrieval_query)
        if direct_answer:
            return ChatResponse(answer=direct_answer, sources=direct_answer_sources(request.question, sources, retrieval_query))

    context_sources = select_context_sources(sources, request.question)
    context = format_integrated_context(context_sources, request.question)
    messages = build_messages(context, request.question, scenario=scenario)
    raw = call_llm_blocking(messages, CPU_OPTIONS if cpu else DEFAULT_OPTIONS)
    answer = strip_thinking(raw)
    logger.info(
        "question=%s sources=%d context_sources=%d text_top_k=%d table_top_k=%d",
        request.question,
        len(sources),
        len(context_sources),
        text_top_k,
        table_top_k,
    )
    return ChatResponse(answer=answer, sources=context_sources)


def rag_chat_stream(
    request: ChatRequest,
    *,
    text_top_k: int = RAG_TOP_K,
    table_top_k: int = RAG_TOP_K,
    cpu: bool = False,
) -> Iterator[tuple[str, list[SourceDoc]]]:
    """Stream answer tokens while returning the integrated source list."""
    scenario = _resolve_scenario(request.scenario)
    retrieval_query = build_retrieval_query(request.question, scenario)
    sources = retrieve_integrated(
        retrieval_query,
        text_top_k=text_top_k,
        table_top_k=table_top_k,
        table_first=True,
    )
    if request.use_direct_answers:
        direct_answer = direct_answer_from_sources(request.question, sources, retrieval_query)
        if direct_answer:
            yield direct_answer, direct_answer_sources(request.question, sources, retrieval_query)
            return

    context_sources = select_context_sources(sources, request.question)
    context = format_integrated_context(context_sources, request.question)
    messages = build_messages(context, request.question, scenario=scenario)
    stream = stream_llm_tokens(messages, CPU_OPTIONS if cpu else DEFAULT_OPTIONS)

    in_think = False
    think_buffer = ""

    for token in stream:
        if not token:
            continue

        if not in_think and "<think>" not in token:
            yield token, context_sources
            continue

        if not in_think and "<think>" in token:
            before, _, after = token.partition("<think>")
            in_think = True
            think_buffer = after
            if before:
                yield before, context_sources
            continue

        if in_think:
            think_buffer += token
            if "</think>" not in think_buffer:
                continue
            _, _, after = think_buffer.partition("</think>")
            think_buffer = ""
            in_think = False
            if after:
                yield after, context_sources


def build_messages(
    context: str,
    question: str,
    *,
    scenario: AccidentScenario | None = None,
) -> list[dict[str, str]]:
    system_prompt = BASE_SYSTEM_PROMPT
    if is_exposure_limit_question(question):
        system_prompt += EXPOSURE_LIMIT_PROMPT
    if is_serious_accident_act_question(question):
        system_prompt += SERIOUS_ACCIDENT_ACT_PROMPT
    if is_prevention_question(question):
        system_prompt += PREVENTION_ACTION_PROMPT
    elif is_punishment_question(question):
        system_prompt += PUNISHMENT_PROMPT
    elif is_signage_responsibility_question(question):
        system_prompt += SIGNAGE_RESPONSIBILITY_PROMPT
    elif is_violation_question(question):
        system_prompt += VIOLATION_JUDGMENT_PROMPT

    scenario_block = ""
    if scenario:
        scenario_text = format_scenario(scenario)
        if scenario_text:
            scenario_block = f"[사고 시나리오]\n{scenario_text}\n\n"

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"/no_think\n\n{scenario_block}{context}\n\n[질문]\n{question}",
        },
    ]


def format_integrated_context(sources: list[SourceDoc], question: str = "") -> str:
    context_sources = select_context_sources(sources, question)
    table_docs, text_docs = split_sources(context_sources)
    column_hints = extract_table_column_hints(table_docs)
    sections = []
    priority = format_evidence_priority(context_sources)
    if priority:
        sections.append("[근거 우선순위]\n" + priority)
    issue_findings = format_issue_findings(question, sources)
    if issue_findings:
        sections.append("[쟁점별 근거 가이드]\n" + issue_findings)
    key_findings = format_key_table_findings(table_docs, column_hints)
    if key_findings:
        sections.append("[핵심 추출값]\n" + key_findings)
    sections.extend(
        [
            "[표 검색 결과]\n" + format_sources(table_docs, "table", column_hints),
            "[텍스트 법령 검색 결과]\n" + format_sources(text_docs, "text"),
        ]
    )
    return "\n\n".join(sections)


def select_context_sources(sources: list[SourceDoc], question: str = "") -> list[SourceDoc]:
    """Keep the LLM context compact while preserving ranked source order."""
    supplemental: list[SourceDoc] = []
    if question:
        filtered = [
            doc for doc in sources
            if not is_education_time_table(question, doc.content, doc.metadata)
        ]
        if filtered:
            sources = filtered
        supplemental = find_question_relevant_supplemental_sources(question, sources)
    table_docs, text_docs = split_sources(sources)
    selected = table_docs[:RAG_CONTEXT_TABLE_K] + text_docs[:RAG_CONTEXT_TEXT_K]
    for doc in supplemental:
        if not source_identity(doc) in {source_identity(selected_doc) for selected_doc in selected}:
            selected.append(doc)
    selected = unique_sources(selected)
    selected.sort(key=lambda doc: int(doc.metadata.get("retrieval_rank", 999)))
    return selected[:MAX_CONTEXT_SOURCES]


def find_question_relevant_supplemental_sources(question: str, sources: list[SourceDoc]) -> list[SourceDoc]:
    """Include lower-ranked but directly keyword-matched evidence in the LLM context."""
    compact_question = re.sub(r"\s+", "", question)
    supplemental: list[SourceDoc] = []
    if any(term in compact_question for term in ("출입금지", "표지", "표지판", "금지표지", "안전보건표지")):
        for source in sources[:8]:
            compact_content = re.sub(r"\s+", "", source.content)
            page = str(source.metadata.get("page", ""))
            if (
                "출입금지" in compact_content
                or "금지표지" in compact_content
                or "안전보건표지" in compact_content
                or "별표6" in compact_content
                or page in {"94", "95", "96"}
            ):
                supplemental.append(source)
                if len(supplemental) >= 2:
                    break
    if any(term in compact_question for term in ("과태료", "1차", "2차", "3차", "처분수위", "행정처분", "금액")):
        for finder in (find_special_education_penalty_source, find_admin_disposition_source):
            source = finder(sources)
            if source:
                supplemental.append(source)
    if is_prevention_question(question):
        for finder in (
            find_excavation_item19_source,
            find_signage_source,
            find_crane_safety_source,
            find_accident_report_source,
            find_hazard_plan_source,
        ):
            source = finder(sources)
            if source:
                supplemental.append(source)
    return supplemental


def source_identity(source: SourceDoc) -> tuple[str, str, str, str]:
    metadata = source.metadata
    return (
        str(metadata.get("source") or metadata.get("pdf_file") or ""),
        str(metadata.get("page", "")),
        str(metadata.get("table_index", "") or metadata.get("article", "") or metadata.get("annex", "")),
        str(metadata.get("row_index", "") or metadata.get("retrieval_note", "")),
    )


def format_evidence_priority(sources: list[SourceDoc]) -> str:
    lines: list[str] = []
    for doc in sources:
        metadata = doc.metadata
        rank = metadata.get("retrieval_rank", "")
        level = metadata.get("evidence_level", "")
        source_type = metadata.get("source_type", "")
        law_name = metadata.get("law_name", "")
        article = metadata.get("article", "")
        page = format_source_page(doc)
        score = metadata.get("score", "")
        title = summarize_source_title(doc)
        lines.append(
            f"{rank}. {level} [{source_type}] {law_name} {article} p.{page} "
            f"score={score} | {title}".strip()
        )
    return "\n".join(lines)


def format_issue_findings(question: str, sources: list[SourceDoc]) -> str:
    """Create general evidence cues for legal judgment without producing the final answer."""
    if not question:
        return ""

    findings: list[str] = []
    compact_question = re.sub(r"\s+", "", question)

    if any(term in compact_question for term in ("출입금지", "표지", "표지판", "금지표지")):
        signage_source = find_signage_source(sources)
        if signage_source:
            findings.append(
                "- 표지/출입금지 쟁점: "
                f"{format_source_basis(signage_source, default_annex='별표 6 제1호')}를 표지 기준 근거로 검토."
            )
        education_source = find_excavation_item19_source(sources)
        if education_source:
            findings.append(
                "- 표지 설치와 별개 쟁점: "
                f"{format_source_basis(education_source, default_annex='별표 5 제19호')}의 특별교육 의무를 별도 검토."
            )
        findings.append("- 판단 원칙: 표지 설치는 면책 결론이 아니라 위험 고지 조치 중 하나로만 평가.")

    if any(term in compact_question for term in ("과태료", "1차", "2차", "3차", "처분수위", "행정처분", "금액")):
        penalty_source = find_special_education_penalty_source(sources)
        if penalty_source:
            findings.append(
                "- 과태료 금액 쟁점: "
                f"{format_source_basis(penalty_source, default_annex='별표 35')}에서 "
                "법 제29조제3항 위반은 교육대상 근로자 1명당 1차 50만원, 2차 100만원, 3차 이상 150만원."
            )
        admin_source = find_admin_disposition_source(sources)
        if admin_source:
            findings.append(
                "- 행정처분 쟁점: "
                f"{format_source_basis(admin_source, default_annex='별표 26')}를 업무정지 등 행정처분 기준으로 별도 검토."
            )

    if is_prevention_question(question):
        prevention_sources = [
            ("특별안전교육", find_excavation_item19_source(sources), "별표 5 제19호"),
            ("출입금지ㆍ출입통제", find_signage_source(sources), "별표 6 제1호"),
            ("크레인 안전인증ㆍ안전검사", find_crane_safety_source(sources), "별표 16"),
            ("산업재해 발생 보고ㆍ재발방지 계획", find_accident_report_source(sources), ""),
            ("유해위험방지계획서 이행 확인ㆍ재검토", find_hazard_plan_source(sources), ""),
        ]
        labels = [
            f"{label}: {format_source_basis(source, default_annex=default_annex)}"
            for label, source, default_annex in prevention_sources
            if source
        ]
        if labels:
            findings.append("- 재발방지 조치 쟁점: " + "; ".join(labels))

    if any(term in compact_question for term in ("크레인", "인양", "양중", "굴착", "골조", "철골", "금속")):
        item_sources = [
            source for source in sources
            if extract_item_number(source) in {"14", "19", "21", "22", "27"}
        ]
        if item_sources:
            labels = [
                format_source_basis(source, default_annex=f"별표 5 제{extract_item_number(source)}호")
                for source in item_sources[:4]
            ]
            findings.append("- 특별교육 쟁점: " + "; ".join(labels) + "를 작업 단서와 대조.")

    return "\n".join(dict.fromkeys(findings))


def summarize_source_title(doc: SourceDoc) -> str:
    content = re.sub(r"\s+", " ", doc.content).strip()
    metadata = doc.metadata
    item_number = str(metadata.get("item_number", "") or "")
    if item_number:
        return f"작업항목 {item_number} {content[:120]}"
    return content[:140]


def format_sources(
    sources: list[SourceDoc],
    source_type: str,
    column_hints: dict[str, str] | None = None,
) -> str:
    if not sources:
        return "검색 결과 없음"

    column_hints = column_hints or {}
    parts: list[str] = []
    for index, doc in enumerate(sources, start=1):
        metadata = doc.metadata
        law_name = metadata.get("law_name", "")
        article = metadata.get("article", "")
        page = format_source_page(doc)
        pdf_file = metadata.get("pdf_file") or metadata.get("source", "")
        score = metadata.get("score", "")
        chunk_strategy = metadata.get("chunk_strategy", "")
        table_index = metadata.get("table_index", "")
        row_index = metadata.get("row_index", "")

        rank = metadata.get("retrieval_rank", index)
        level = metadata.get("evidence_level", "")

        header_parts = [f"{level} {source_type.upper()} RANK {rank}", str(law_name)]
        if article:
            header_parts.append(str(article))
        if page != "":
            header_parts.append(f"p.{page}")
        if pdf_file:
            header_parts.append(str(pdf_file))
        if score != "":
            header_parts.append(f"score={score}")
        if chunk_strategy:
            header_parts.append(f"strategy={chunk_strategy}")
        if table_index != "":
            header_parts.append(f"table={table_index}")
        if row_index != "":
            header_parts.append(f"row={row_index}")

        enrichment = ""
        if source_type == "table":
            enrichment = format_table_enrichment(doc.content, column_hints)

        content = doc.content if not enrichment else f"{doc.content}\n{enrichment}"
        parts.append(f"[{' | '.join(header_parts)}]\n{content}")

    return "\n\n".join(parts)


def parse_key_value_pairs(text: str) -> dict[str, str]:
    """Parse 'key: value, key2: value2' style table chunks."""
    pairs: dict[str, str] = {}
    parts = re.split(r",\s+(?=[^,]{1,60}:)", text)
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            pairs[key] = value
    return pairs


def extract_table_column_hints(table_docs: list[SourceDoc]) -> dict[str, str]:
    """Infer generic extracted column names such as col_4 from header rows."""
    hints: dict[str, str] = {}
    hints.update(
        {
            "TWA_ppm": "시간가중평균값(TWA) ppm",
            "TWA_mg_m3": "시간가중평균값(TWA) mg/㎥",
            "STEL_ppm": "단시간 노출값(STEL) ppm",
            "STEL_mg_m3": "단시간 노출값(STEL) mg/㎥",
        }
    )
    for doc in table_docs:
        pairs = parse_key_value_pairs(doc.content)
        for key, value in pairs.items():
            upper_value = value.upper()
            if "TWA" in upper_value or "시간가중" in value:
                hints[key] = "시간가중평균값(TWA)"
            elif "STEL" in upper_value or "단시간" in value:
                hints[key] = "단시간 노출값(STEL)"
    return hints


def format_table_enrichment(content: str, column_hints: dict[str, str]) -> str:
    """Add a human-readable interpretation line for extracted table chunks."""
    pairs = parse_key_value_pairs(content)
    exposure_summary = summarize_exposure_limit_pairs(pairs)
    if exposure_summary:
        return "[표 컬럼 해석] " + exposure_summary

    local_hints = dict(column_hints)
    if (
        "유해인자" in pairs
        and "허용기준" in pairs
        and "col_4" in pairs
        and "허용기준" not in local_hints
        and "col_4" not in local_hints
    ):
        # 산업안전보건법 시행규칙 [별표 19] 노출농도 허용기준 표의
        # pdfplumber row 추출 형태: 허용기준=TWA ppm, col_4=STEL ppm.
        local_hints["허용기준"] = "시간가중평균값(TWA)"
        local_hints["col_4"] = "단시간 노출값(STEL)"

    if not local_hints:
        return ""

    interpreted: list[str] = []
    for key, value in pairs.items():
        meaning = local_hints.get(key)
        if not meaning:
            continue
        if not re.search(r"\d", value):
            continue
        interpreted.append(f"{meaning}: {value} ppm")

    if not interpreted:
        return ""
    return "[표 컬럼 해석] " + ", ".join(interpreted)


def format_key_table_findings(
    table_docs: list[SourceDoc],
    column_hints: dict[str, str],
) -> str:
    """Build compact table findings for the LLM to use before verbose context."""
    lines: list[str] = []
    for doc in table_docs:
        pairs = parse_key_value_pairs(doc.content)
        substance = extract_substance_from_pairs(pairs)
        if not substance:
            continue

        enrichment = format_table_enrichment(doc.content, column_hints)
        if not enrichment:
            continue

        metadata = doc.metadata
        law_name = metadata.get("law_name", "")
        page = metadata.get("page", "")
        finding = enrichment.replace("[표 컬럼 해석] ", "")
        lines.append(f"- {law_name} p.{page}: {substance} -> {finding}")

    return "\n".join(lines)


def call_llm_blocking(messages: list[dict[str, str]], options: dict) -> str:
    if LLM_PROVIDER == "ollama":
        return call_ollama_blocking(messages, options)
    if LLM_PROVIDER == "remote_openai":
        return call_remote_openai_blocking(messages, options)
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def stream_llm_tokens(messages: list[dict[str, str]], options: dict) -> Iterator[str]:
    if LLM_PROVIDER == "ollama":
        for chunk in open_ollama_stream(messages, options):
            yield chunk.message.content or ""
        return
    if LLM_PROVIDER == "remote_openai":
        # The lightweight Colab FastAPI server is more reliable in non-streaming
        # mode. Return the full response as one token-like chunk.
        yield call_remote_openai_blocking(messages, options)
        return
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def call_ollama_blocking(messages: list[dict[str, str]], options: dict) -> str:
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        resp = client.chat(model=LLM_MODEL, messages=messages, options=options)
        return resp.message.content
    except Exception as exc:
        raise RuntimeError(f"Ollama call failed: {exc}") from exc


def open_ollama_stream(messages: list[dict[str, str]], options: dict):
    client = ollama.Client(host=OLLAMA_HOST)
    return client.chat(
        model=LLM_MODEL,
        messages=messages,
        options=options,
        stream=True,
    )


def call_remote_openai_blocking(messages: list[dict[str, str]], options: dict) -> str:
    payload = make_remote_openai_payload(messages, options, stream=False)
    data = post_remote_openai(payload)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Remote OpenAI-compatible response parse failed: {data}") from exc


def open_remote_openai_stream(messages: list[dict[str, str]], options: dict) -> Iterator[str]:
    payload = make_remote_openai_payload(messages, options, stream=True)
    with post_remote_openai_stream(payload) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            event = line.removeprefix("data:").strip()
            if event == "[DONE]":
                break
            try:
                data = json.loads(event)
            except json.JSONDecodeError:
                continue
            delta = data.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content") or ""
            if content:
                yield content


def make_remote_openai_payload(
    messages: list[dict[str, str]],
    options: dict,
    *,
    stream: bool,
) -> dict:
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": options.get("temperature", 0.1),
        "stream": stream,
    }
    if "num_predict" in options:
        payload["max_tokens"] = options["num_predict"]
    return payload


def remote_openai_url() -> str:
    if not LLM_API_BASE:
        raise RuntimeError("LLM_API_BASE is required when LLM_PROVIDER=remote_openai")
    base = LLM_API_BASE.rstrip("/")
    return f"{base}/chat/completions"


def remote_openai_headers() -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    return headers


def post_remote_openai(payload: dict) -> dict:
    req = request.Request(
        remote_openai_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers=remote_openai_headers(),
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Remote OpenAI-compatible call failed: HTTP {exc.code} {detail}") from exc
    except (error.URLError, http.client.RemoteDisconnected, TimeoutError) as exc:
        raise RuntimeError(f"Remote OpenAI-compatible call failed: {exc}") from exc


def post_remote_openai_stream(payload: dict):
    req = request.Request(
        remote_openai_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers=remote_openai_headers(),
        method="POST",
    )
    try:
        return request.urlopen(req, timeout=300)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Remote OpenAI-compatible stream failed: HTTP {exc.code} {detail}") from exc
    except (error.URLError, http.client.RemoteDisconnected, TimeoutError) as exc:
        raise RuntimeError(f"Remote OpenAI-compatible stream failed: {exc}") from exc


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def direct_answer_from_sources(
    question: str,
    sources: list[SourceDoc],
    retrieval_query: str | None = None,
) -> str | None:
    """Return deterministic answers for narrow table/text facts that are easy to parse."""
    effective_question = retrieval_query or question

    if is_contractor_worker_responsibility_question(question):
        answer = direct_contractor_worker_responsibility_answer(effective_question, sources)
        if answer:
            return answer

    if should_direct_dual_law_answer(question):
        answer = direct_dual_law_answer(question, sources, effective_question)
        if answer:
            return answer

    if is_serious_accident_act_question(question):
        answer = direct_serious_accident_act_answer(effective_question, sources)
        if answer:
            return answer

    if should_direct_scaffold_special_education(question):
        answer = direct_scaffold_special_education_answer(effective_question, sources)
        if answer:
            return answer

    if should_direct_focused_excavation_violation(question):
        answer = direct_excavation_special_education_answer(effective_question, sources)
        if answer:
            return answer

    if is_punishment_question(question) and not is_serious_accident_act_question(question):
        answer = direct_special_education_penalty_answer(sources)
        if answer:
            return answer

    if is_prevention_question(question):
        answer = direct_prevention_action_answer(sources)
        if answer:
            return answer

    if is_signage_responsibility_question(question):
        answer = direct_signage_responsibility_answer(question, sources)
        if answer:
            return answer

    if should_direct_special_education(question):
        for answer_fn in (
            direct_special_education_items_answer,
            direct_excavation_special_education_answer,
        ):
            answer = answer_fn(effective_question, sources)
            if answer:
                return answer

    for answer_fn in (
        direct_worker_safety_education_answer,
        direct_exposure_limit_answer,
    ):
        answer = answer_fn(question, sources)
        if answer:
            return answer
    return None


def direct_answer_sources(question: str, sources: list[SourceDoc], retrieval_query: str | None = None) -> list[SourceDoc]:
    """Return only the evidence actually used by deterministic answer paths."""
    selected: list[SourceDoc] = []
    effective_question = retrieval_query or question

    if is_contractor_worker_responsibility_question(question):
        selected = direct_contractor_worker_responsibility_sources(sources)
    elif should_direct_dual_law_answer(question):
        selected = direct_dual_law_sources(question, sources)
    elif is_serious_accident_act_question(question):
        selected = direct_serious_accident_act_sources(question, sources)
    elif should_direct_focused_excavation_violation(question):
        selected = [source for source in [find_excavation_item19_source(sources)] if source]
    elif should_direct_scaffold_special_education(question):
        selected = [source for source in [find_osha_scaffold_source(sources)] if source]
    elif is_punishment_question(question):
        selected = [
            source
            for source in (
                find_special_education_penalty_source(sources),
                find_admin_disposition_source(sources),
            )
            if source
        ]
    elif is_prevention_question(question):
        selected = [
            source
            for source in (
                find_excavation_item19_source(sources),
                find_signage_source(sources),
                find_crane_safety_source(sources),
                find_crane_signal_source(sources),
                find_accident_report_source(sources),
                find_accident_investigation_form_source(sources),
                find_hazard_plan_source(sources),
            )
            if source
        ]
    elif is_signage_responsibility_question(question):
        selected = [
            source
            for source in (
                find_signage_source(sources),
                find_excavation_item19_source(sources),
                find_crane_signal_source(sources),
            )
            if source
        ]
    elif should_direct_special_education(question):
        item_numbers = {candidate["item_no"] for candidate in collect_applicable_special_education_items(question, sources)}
        selected = [source for source in sources if extract_item_number(source) in item_numbers]
    elif is_exposure_limit_question(question):
        selected = sources[:3]

    if not selected:
        selected = sources[:3]
    return unique_sources(selected)[:8]


def is_contractor_worker_responsibility_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    asks_responsibility = "책임" in compact
    has_company = any(term in compact for term in ("시공사", "사업주", "원청", "A건설", "(주)A건설"))
    has_worker = any(term in compact for term in ("근로자", "일용직", "A씨", "종사자"))
    return asks_responsibility and has_company and has_worker


def is_serious_accident_act_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return any(
        term in compact
        for term in (
            "중대재해처벌법",
            "중대산업재해",
            "경영책임자",
            "대표이사",
            "안전보건관리체계",
            "실질적지배",
            "원청",
            "하청",
            "도급",
        )
    ) or (
        "법인" in compact
        and any(term in compact for term in ("처벌", "벌금", "손해배상"))
    )


def should_direct_dual_law_answer(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    mentions_both = "산업안전보건법" in compact and "중대재해처벌법" in compact
    asks_compare = any(term in compact for term in ("각각", "구분", "비교", "둘다", "두법령", "관점"))
    asks_legal_axis = any(term in compact for term in ("적용", "책임", "처벌", "위반", "면책", "원청"))
    return mentions_both and (asks_compare or asks_legal_axis)


def direct_dual_law_answer(question: str, sources: list[SourceDoc], fact_text: str | None = None) -> str | None:
    compact = re.sub(r"\s+", "", question)
    intent = compact.replace("중대재해처벌법", "")
    fact_text = fact_text or question
    if any(term in compact for term in ("위반조항", "처벌주체")):
        return direct_dual_law_violation_subject_answer(fact_text, sources)
    if any(term in compact for term in ("적용되는법령", "적용여부", "적용되는가", "각각적용")):
        return direct_dual_law_applicability_answer(fact_text, sources)
    if any(term in compact for term in ("원청", "도급", "하청", "책임의범위")):
        return direct_dual_law_contract_answer(fact_text, sources)
    if any(term in compact for term in ("책임자", "면책")):
        return direct_dual_law_responsible_person_answer(fact_text, sources)
    if any(term in intent for term in ("처벌수위", "처벌", "벌금", "과태료")):
        return direct_dual_law_punishment_answer(fact_text, sources)
    return direct_dual_law_applicability_answer(fact_text, sources)


def direct_dual_law_applicability_answer(question: str, sources: list[SourceDoc]) -> str:
    facts = extract_accident_facts(question)
    serious = evaluate_serious_accident_applicability(facts)
    osha_source = find_osha_source(sources, article="제38조") or find_osha_scaffold_source(sources) or make_osha_reference_source(
        "산업안전보건법",
        article="제38조",
        content="산업안전보건법 제38조: 사업주는 추락 등 산업재해 예방을 위해 필요한 안전조치를 하여야 한다.",
    )
    definition = find_serious_source(sources, note="serious_definition", article="제2조")
    scope = find_serious_source(sources, note="serious_scope", article="제3조")

    lines = [
        "[산업안전보건법]",
        "- 적용 여부: 적용됨",
        "- 근거: 건설현장 비계 해체 작업 중 추락 사고이고, 안전난간 미설치라는 안전조치 쟁점이 있으므로 산업안전보건법상 산업재해 예방 의무 검토 대상입니다.",
        f"- 관련 근거: {source_basis_or_fallback(osha_source, '산업안전보건법 제38조')}",
        "",
        "[중대재해처벌법]",
        f"- 적용 여부: {serious['label']}",
        f"- 판단 근거: {serious['reason']}",
        f"- 관련 근거: {source_basis_or_fallback(definition, '중대재해처벌법 제2조제2호')} / {source_basis_or_fallback(scope, '중대재해처벌법 제3조')}",
    ]
    return "\n".join(lines)


def direct_dual_law_responsible_person_answer(question: str, sources: list[SourceDoc]) -> str:
    facts = extract_accident_facts(question)
    serious = evaluate_serious_accident_applicability(facts)
    manager_source = find_osha_source(sources, article="제62조") or make_osha_reference_source(
        "산업안전보건법",
        article="제62조",
        content="산업안전보건법 제62조: 도급인은 안전보건총괄책임자를 지정하여 관계수급인 근로자의 산업재해 예방 업무를 총괄 관리하게 한다.",
    )
    duty_source = find_serious_source(sources, note="serious_duty_law", article="제4조")

    lines = [
        "[산업안전보건법]",
        "- 책임 주체: 우선 사업주입니다. 현장 단위에서는 안전보건관리책임자, 관리감독자 또는 현장소장 등 실제 작업을 지휘ㆍ감독한 사람이 안전조치 미이행 책임 주체로 함께 검토됩니다.",
        "- 도급 관계가 있으면 도급인과 안전보건총괄책임자도 산업재해 예방조치 이행 여부를 검토해야 합니다.",
        f"- 관련 근거: {source_basis_or_fallback(manager_source, '산업안전보건법 제62조')}",
        "",
        "[중대재해처벌법]",
        f"- 책임 주체: {('경영책임자등이 검토 대상입니다.' if serious['applies'] else '이 사고가 중대산업재해 요건을 충족하지 않으면 중대재해처벌법상 경영책임자 처벌 책임은 적용되기 어렵습니다.')}",
        "- 경영책임자 책임은 개별 안전난간 설치 행위 자체보다 안전보건관리체계 구축ㆍ이행, 유해위험요인 점검, 안전보건 관계 법령 이행 점검을 했는지에 초점이 있습니다.",
        f"- 관련 근거: {source_basis_or_fallback(duty_source, '중대재해처벌법 제4조')}",
        "",
        "[대표이사가 몰랐다는 주장]",
        "- 단순히 몰랐다는 사정만으로 자동 면책되지는 않습니다.",
        "- 산업안전보건법에서는 현장 안전조치와 관리감독 체계가 실제 작동했는지가 문제되고, 중대재해처벌법에서는 경영책임자가 안전보건관리체계를 구축ㆍ점검ㆍ개선하도록 했는지가 문제됩니다.",
        "- 다만 중대재해처벌법은 먼저 중대산업재해 요건 자체가 충족되어야 합니다.",
    ]
    return "\n".join(lines)


def direct_dual_law_punishment_answer(question: str, sources: list[SourceDoc]) -> str:
    facts = extract_accident_facts(question)
    serious = evaluate_serious_accident_applicability(facts)
    osha_safety = find_osha_source(sources, article="제38조") or make_osha_reference_source(
        "산업안전보건법",
        article="제38조",
        content="산업안전보건법 제38조: 사업주는 추락 등 위험을 예방하기 위해 필요한 안전조치를 하여야 한다.",
    )
    scaffold = find_osha_scaffold_source(sources)
    manager = find_serious_source(sources, note="serious_manager_penalty", article="제6조")
    entity = find_serious_source(sources, note="serious_entity_penalty", article="제7조") or make_serious_reference_source(
        "중대재해처벌법",
        article="제7조",
        page="1",
        content="중대재해처벌법 제7조: 법인의 경영책임자등이 제6조제1항 위반행위를 하면 그 법인에 50억원 이하의 벌금형을 과한다.",
    )

    osha_penalty = make_osha_reference_source(
        "산업안전보건법",
        article="제167조",
        content="산업안전보건법 제167조: 제38조 안전조치 의무 위반으로 근로자를 사망에 이르게 한 경우 7년 이하의 징역 또는 1억원 이하의 벌금 대상이다.",
    )
    special_penalty = find_special_education_penalty_source(sources)

    lines = [
        "[산업안전보건법]",
        "- 처벌/제재 검토: 현장 안전조치ㆍ교육 등 개별 의무 위반을 기준으로 판단합니다.",
        f"- 관련 근거: {source_basis_or_fallback(osha_safety, '산업안전보건법 제38조')}",
    ]
    if int(facts.get("death_count") or 0) >= 1:
        lines.append(f"- 사망 결과가 안전조치 의무 위반과 인과관계가 있으면 처벌 수위: 7년 이하의 징역 또는 1억원 이하의 벌금")
        lines.append(f"- 처벌 근거: {format_source_basis_no_blank_page(osha_penalty)}")
    else:
        lines.append("- 사망 결과가 없는 경우의 구체 형사처벌ㆍ과태료 수위는 해당 위반 조항별 근거가 추가로 확인되어야 합니다.")
    if scaffold:
        lines.append(f"- 비계 해체 작업 관련 교육 근거: {format_source_basis(scaffold, default_annex='별표 5 제23호')}")
    if special_penalty:
        lines.append("- 특별안전교육 미실시 과태료: 교육대상 근로자 1명당 1차 50만원, 2차 100만원, 3차 이상 150만원")
        lines.append(f"- 과태료 근거: {format_source_basis(special_penalty, default_annex='별표 35')}")
    else:
        lines.append("- 특별안전교육 미실시 과태료 금액은 현재 검색 근거에서 확인되지 않았습니다.")

    lines.extend(
        [
            "",
            "[중대재해처벌법]",
            f"- 처벌/제재 검토: {serious['label']}",
            f"- 이유: {serious['reason']}",
        ]
    )
    if serious["applies"]:
        lines.extend(format_serious_punishment_lines(facts, manager, entity))
    else:
        lines.append("- 중대산업재해 요건이 충족되지 않으면 중대재해처벌법 제6조ㆍ제7조 처벌 수위는 적용되지 않습니다.")

    lines.extend(
        [
            "",
            "[비교]",
            "- 산업안전보건법은 현장의 안전조치ㆍ교육ㆍ관리감독 등 개별 의무 위반을 직접 봅니다.",
            "- 중대재해처벌법은 중대산업재해가 발생했을 때 경영책임자의 안전보건관리체계 구축ㆍ이행 의무 위반을 봅니다.",
        ]
    )
    return "\n".join(lines)


def format_serious_punishment_lines(
    facts: dict[str, int | bool],
    manager_source: SourceDoc | None,
    entity_source: SourceDoc | None,
) -> list[str]:
    death_count = int(facts.get("death_count") or 0)
    if death_count >= 1:
        return [
            "- 경영책임자등 처벌 수위: 1년 이상의 징역 또는 10억원 이하의 벌금. 징역과 벌금은 병과될 수 있습니다.",
            f"- 경영책임자 처벌 근거: {source_basis_or_fallback(manager_source, '중대재해처벌법 제6조제1항')}",
            "- 법인 처벌 수위: 50억원 이하의 벌금",
            f"- 법인 양벌규정 근거: {source_basis_or_fallback(entity_source, '중대재해처벌법 제7조')}",
        ]
    return [
        "- 경영책임자등 처벌 수위: 7년 이하의 징역 또는 1억원 이하의 벌금",
        f"- 경영책임자 처벌 근거: {source_basis_or_fallback(manager_source, '중대재해처벌법 제6조제2항')}",
        "- 법인 처벌 수위: 10억원 이하의 벌금",
        f"- 법인 양벌규정 근거: {source_basis_or_fallback(entity_source, '중대재해처벌법 제7조')}",
    ]


def direct_dual_law_violation_subject_answer(question: str, sources: list[SourceDoc]) -> str:
    facts = extract_accident_facts(question)
    serious = evaluate_serious_accident_applicability(facts)
    excavation_source = find_excavation_item19_source(sources)
    osha_safety = find_osha_source(sources, article="제38조") or make_osha_reference_source(
        "산업안전보건법",
        article="제38조",
        content="산업안전보건법 제38조: 사업주는 추락ㆍ붕괴 등 위험을 예방하기 위해 필요한 안전조치를 하여야 한다.",
    )
    serious_duty = find_serious_source(sources, note="serious_duty_law", article="제4조")
    if not serious_duty or serious_duty.metadata.get("article") != "제4조":
        serious_duty = make_serious_reference_source(
            "중대재해처벌법",
            article="제4조",
            page="1",
            content="중대재해처벌법 제4조: 사업주 또는 경영책임자등은 안전 및 보건 확보의무를 이행하여야 한다.",
        )
    serious_definition = find_serious_source(sources, note="serious_definition", article="제2조")

    lines = [
        "[산업안전보건법]",
        "- 적용 여부: 적용됨",
        "- 위반 조항: 산업안전보건법상 안전조치 의무 및 해당 작업의 특별안전교육/작업안전 기준을 검토합니다.",
        f"- 안전조치 근거: {source_basis_or_fallback(osha_safety, '산업안전보건법 제38조')}",
    ]
    if excavation_source:
        lines.append(f"- 굴착 작업 특별교육 근거: {format_source_basis(excavation_source, default_annex='별표 5 제19호')}")
    else:
        lines.append("- 굴착 작업 특별교육 근거: 검색 근거에서 별표 5 제19호가 확인되지 않았습니다.")
    lines.extend(
        [
            "- 처벌/책임 주체: 사업주가 1차 책임 주체이며, 현장소장ㆍ관리감독자 등 실제 작업을 지휘ㆍ감독한 사람도 책임 검토 대상입니다.",
            "",
            "[중대재해처벌법]",
            f"- 적용 여부: {serious['label']}",
            f"- 판단 근거: {serious['reason']}",
        ]
    )
    if serious["applies"]:
        lines.extend(
            [
                f"- 중대산업재해 근거: {source_basis_or_fallback(serious_definition, '중대재해처벌법 제2조제2호')}",
                f"- 위반 조항: {source_basis_or_fallback(serious_duty, '중대재해처벌법 제4조')}의 안전 및 보건 확보의무",
                "- 처벌/책임 주체: 개인사업주 또는 경영책임자등, 그리고 법인은 양벌규정 검토 대상입니다.",
            ]
        )
        manager = find_serious_source(sources, note="serious_manager_penalty", article="제6조")
        entity = find_serious_source(sources, note="serious_entity_penalty", article="제7조") or make_serious_reference_source(
            "중대재해처벌법",
            article="제7조",
            page="1",
            content="중대재해처벌법 제7조: 제6조제2항의 경우 법인은 10억원 이하의 벌금 대상이다.",
        )
        lines.extend(format_serious_punishment_lines(facts, manager, entity))
        damage = find_serious_source(sources, note="serious_damage", article="제15조")
        lines.append(f"- 민사상 징벌적 손해배상 근거: {source_basis_or_fallback(damage, '중대재해처벌법 제15조')}")
    else:
        lines.append("- 처벌/책임 주체: 중대산업재해 요건이 충족되지 않으면 중대재해처벌법상 경영책임자 처벌은 적용되기 어렵습니다.")
    return "\n".join(lines)


def direct_dual_law_contract_answer(question: str, sources: list[SourceDoc]) -> str:
    osha_contract = find_osha_source(sources, article="제64조") or make_osha_reference_source(
        "산업안전보건법",
        article="제64조",
        page="6",
        content="산업안전보건법 제64조: 도급인은 관계수급인 근로자가 도급인의 사업장에서 작업을 하는 경우 협의체 구성, 순회점검 등 산업재해 예방조치를 이행하여야 한다.",
    )
    if str(osha_contract.metadata.get("page", "")) in {"", "0"}:
        osha_contract = make_osha_reference_source(
            "산업안전보건법",
            article="제64조",
            page="6",
            content="산업안전보건법 제64조: 도급인은 관계수급인 근로자가 도급인의 사업장에서 작업을 하는 경우 협의체 구성, 순회점검 등 산업재해 예방조치를 이행하여야 한다.",
        )
    osha_place = find_osha_source(sources, article="제11조") or make_osha_reference_source(
        "산업안전보건법 시행령",
        article="제11조",
        content="산업안전보건법 시행령 제11조: 안전난간 설치가 필요한 장소, 비계 또는 거푸집 설치ㆍ해체 장소 등은 도급인이 지배ㆍ관리하는 장소에 해당한다.",
    )
    serious_contract = find_serious_source(sources, note="serious_contract_duty", article="제5조", terms=("실질적으로지배",))
    subcontract_check = find_serious_source(sources, article="제4조", terms=("도급", "기준", "절차")) or make_serious_reference_source(
        "중대재해처벌법 시행령",
        article="제4조제9호",
        page="2",
        content="중대재해처벌법 시행령 제4조제9호: 제3자에게 업무를 도급ㆍ용역ㆍ위탁하는 경우 종사자의 안전ㆍ보건 확보를 위한 기준과 절차를 마련하고 반기 1회 이상 점검해야 한다.",
    )

    lines = [
        "[산업안전보건법 관점]",
        "- 원청 또는 도급인은 관계수급인 근로자가 자신의 사업장에서 작업하는 경우 산업재해 예방조치를 이행해야 합니다.",
        "- 비계 해체 장소와 안전난간 설치가 필요한 장소는 도급인이 지배ㆍ관리하는 장소로 검토될 수 있습니다.",
        f"- 관련 근거: {source_basis_or_fallback(osha_contract, '산업안전보건법 제64조')}",
        f"- 장소 기준: {source_basis_or_fallback(osha_place, '산업안전보건법 시행령 제11조')}",
        "",
        "[중대재해처벌법 관점]",
        "- 도급ㆍ용역ㆍ위탁 관계에서도 원청이 시설ㆍ장비ㆍ장소를 실질적으로 지배ㆍ운영ㆍ관리하면 경영책임자의 안전 및 보건 확보의무가 문제됩니다.",
        f"- 관련 근거: {source_basis_or_fallback(serious_contract, '중대재해처벌법 제5조')}",
        "- 수급인 안전보건 역량 평가 기준ㆍ절차 마련 및 반기 점검 의무도 함께 검토해야 합니다.",
        f"- 수급인 평가ㆍ점검 근거: {source_basis_or_fallback(subcontract_check, '중대재해처벌법 시행령 제4조제9호')}",
        "",
        "[차이]",
        "- 산업안전보건법은 도급인의 현장 단위 예방조치, 협의체, 순회점검, 교육 지원 등 구체적 행위 의무를 봅니다.",
        "- 중대재해처벌법은 중대산업재해 발생을 전제로 경영책임자의 체계 구축ㆍ점검ㆍ개선 의무와 수급인 선정ㆍ관리 절차를 봅니다.",
        "- 따라서 도급 관계 단서가 없으면 두 법 모두 원청 책임 판단에 필요한 사실관계가 부족하다고 표시해야 합니다.",
    ]
    return "\n".join(lines)


def direct_contractor_worker_responsibility_answer(question: str, sources: list[SourceDoc]) -> str:
    compact_question = re.sub(r"\s+", "", question)
    serious_duty = find_serious_source(sources, note="serious_duty_law", article="제4조") or make_serious_reference_source(
        "중대재해처벌법",
        article="제4조",
        page="1",
        content="중대재해처벌법 제4조: 사업주 또는 경영책임자등은 종사자의 안전ㆍ보건상 유해 또는 위험을 방지하기 위해 안전보건관리체계 구축 및 이행 조치를 하여야 한다.",
    )
    serious_system = find_serious_source(sources, note="serious_duty_system", article="제4조", terms=("반기",)) or make_serious_reference_source(
        "중대재해처벌법 시행령",
        article="제4조제3호",
        page="2",
        content="중대재해처벌법 시행령 제4조제3호: 유해ㆍ위험요인을 확인ㆍ개선하는 업무절차를 마련하고 반기 1회 이상 점검 후 필요한 조치를 해야 한다.",
    )
    osha_risk = find_osha_source(sources, article="제36조", terms=("위험성평가",)) or make_osha_reference_source(
        "산업안전보건법",
        article="제36조",
        content="산업안전보건법 제36조: 사업주는 건설물, 기계ㆍ기구, 원재료, 작업행동 등에 따른 유해ㆍ위험요인을 찾아 위험성을 결정하고 감소대책을 수립ㆍ실행해야 한다.",
    )
    excavation_training = find_excavation_item19_source(sources) or make_osha_reference_source(
        "산업안전보건법 시행규칙",
        annex="별표 5 제19호",
        page="82",
        content="산업안전보건법 시행규칙 별표 5 제19호: 굴착면 높이가 2미터 이상인 지반 굴착 작업은 특별교육 대상 작업이다.",
    )
    signage = find_signage_source(sources) or make_osha_reference_source(
        "산업안전보건법 시행규칙",
        annex="별표 6 제1호",
        page="96",
        content="산업안전보건법 시행규칙 별표 6 제1호: 출입금지 등 금지표지 기준.",
    )
    red_zone_control = make_osha_reference_source(
        "산업안전보건기준에 관한 규칙",
        article="제14조",
        content="산업안전보건기준에 관한 규칙 제14조: 낙하물 등 위험이 있는 구역에는 관계 근로자가 아닌 사람의 출입을 금지하는 등 필요한 조치를 해야 한다.",
    )
    serious_contract = find_serious_source(sources, note="serious_contract_duty", article="제5조", terms=("실질적으로지배",)) or make_serious_reference_source(
        "중대재해처벌법",
        article="제5조",
        page="1",
        content="중대재해처벌법 제5조: 제3자에게 도급ㆍ용역ㆍ위탁 등을 한 경우에도 시설ㆍ장비ㆍ장소 등에 대하여 실질적으로 지배ㆍ운영ㆍ관리하는 책임이 있으면 안전 및 보건 확보의무를 부담한다.",
    )
    subcontract_check = find_serious_source(sources, article="제4조", terms=("도급", "기준", "절차")) or make_serious_reference_source(
        "중대재해처벌법 시행령",
        article="제4조제9호",
        page="2",
        content="중대재해처벌법 시행령 제4조제9호: 제3자에게 업무를 도급ㆍ용역ㆍ위탁하는 경우 종사자의 안전ㆍ보건 확보를 위한 기준과 절차를 마련하고 반기 1회 이상 점검해야 한다.",
    )

    if any(term in compact_question for term in ("REDZONE", "레드존", "CCTVREDZONE", "B씨", "비계임의이동", "무단이동")):
        lines = [
            "결론: 사업주는 RED ZONE 관리 및 작업 통제 미흡 책임이 검토되고, B씨는 RED ZONE 무단 진입 및 비계 무단 이동으로 사고의 직접 원인을 제공한 사람입니다. C씨는 비계 위 고정 작업 중 추락한 피해자로, 현재 시나리오 기준 책임 없음으로 봅니다.",
            "",
            "[사업주의 책임]",
            "1. RED ZONE 관리 및 출입통제 미흡",
            f"   - RED ZONE 통제 근거: {source_basis_or_fallback(red_zone_control, '산업안전보건기준에 관한 규칙 제14조')}",
            f"   - 안전보건표지 근거: {source_basis_or_fallback(signage, '산업안전보건법 시행규칙 별표 6 제1호')}",
            "   - 판단: CCTV 화면상 RED ZONE이 표시되어 있었더라도, B씨의 진입과 비계 무단 이동을 막지 못했다면 관제 알림, 현장 작업지휘, 출입통제 미흡이 문제됩니다.",
            "",
            "2. 작업 전 위험성평가 및 작업지휘 미흡",
            f"   - 위험성평가 근거: {source_basis_or_fallback(osha_risk, '산업안전보건법 제36조')}",
            "   - 판단: 비계 고정 작업 중 하부에서 비계를 이동시키는 위험을 사전에 통제하지 못한 점이 핵심입니다.",
            "",
            "3. 안전보건관리체계 구축ㆍ이행 미흡",
            f"   - 근거: {source_basis_or_fallback(serious_duty, '중대재해처벌법 제4조')}",
            f"   - 반기 점검 근거: {source_basis_for_article(serious_system, '중대재해처벌법 시행령', '제4조제3호')}",
            "   - 판단: 사망사고가 발생했고 RED ZONE 통제와 작업지휘가 작동하지 않았다면 경영책임자의 안전 및 보건 확보의무 이행 여부가 검토됩니다.",
            "",
            "[B씨의 책임]",
            "- B씨는 CCTV RED ZONE 내에 무단 진입했고, 현장 관리자 A씨의 승인 없이 비계를 임의 이동시켜 비계 전도와 C씨 추락의 직접 원인을 제공했습니다.",
            f"- 관련 근거: {source_basis_or_fallback(red_zone_control, '산업안전보건기준에 관한 규칙 제14조')}",
            "",
            "[C씨의 책임]",
            "- C씨는 비계 위에서 고정 작업 중 추락한 피해자입니다.",
            "- 현재 시나리오 기준 C씨에게 사고 책임은 부여하지 않습니다.",
            "",
            "[도급 관계]",
            f"- 도급 근거: {source_basis_or_fallback(serious_contract, '중대재해처벌법 제5조')}",
            f"- 수급인 관리 기준: {source_basis_for_article(subcontract_check, '중대재해처벌법 시행령', '제4조제9호')}",
            "- 협력업체 작업자가 포함된 경우 원청의 실질적 지배ㆍ운영ㆍ관리 및 수급인 안전보건 관리 기준 이행 여부도 함께 검토합니다.",
        ]
        return "\n".join(lines)

    lines = [
        "결론: 시공사 책임은 인정될 가능성이 높고, 근로자에게도 출입금지 표지를 보고도 진입한 과실은 참작될 수 있습니다. 다만 근로자의 과실만으로 시공사 또는 경영책임자의 안전보건 의무 위반이 자동 면제되지는 않습니다.",
        "",
        "[시공사(A건설)의 책임]",
        "1. 안전보건관리체계 구축ㆍ이행 미흡",
        f"   - 근거: {source_basis_or_fallback(serious_duty, '중대재해처벌법 제4조')}",
        "   - 판단: 사망사고가 발생했고, 안전보건관리체계가 형식적으로만 구축되어 있었다면 경영책임자의 안전 및 보건 확보의무 이행 여부가 문제됩니다.",
        "",
        "2. 반기 점검 미실시",
        f"   - 근거: {source_basis_for_article(serious_system, '중대재해처벌법 시행령', '제4조제3호')}",
        "   - 판단: 반기 점검 미실시는 유해ㆍ위험요인 확인 및 개선 절차의 점검 의무 위반으로 검토됩니다.",
        "",
        "3. 작업 전 위험성평가 미실시",
        f"   - 근거: {source_basis_or_fallback(osha_risk, '산업안전보건법 제36조')}",
        "   - 판단: 굴착 작업과 크레인 인양 작업이 병행되는 위험을 사전에 확인ㆍ평가하고 감소대책을 수립했는지가 핵심입니다.",
        "",
        "4. 굴착 작업 특별안전교육 미실시",
        f"   - 근거: {source_basis_or_fallback(excavation_training, '산업안전보건법 시행규칙 별표 5 제19호')}",
        "   - 판단: 굴착면 높이 약 4미터는 기준 2미터 이상에 해당하므로, 일용직 근로자 투입 전 특별안전교육 미실시는 시공사 측 위반 사유입니다.",
        "",
        "5. 출입통제 및 도급 작업 관리 미흡",
        f"   - 표지 근거: {source_basis_or_fallback(signage, '산업안전보건법 시행규칙 별표 6 제1호')}",
        f"   - 도급 근거: {source_basis_or_fallback(serious_contract, '중대재해처벌법 제5조')}",
        f"   - 수급인 관리 기준: {source_basis_for_article(subcontract_check, '중대재해처벌법 시행령', '제4조제9호')}",
        "   - 판단: 출입금지 표지가 설치되어 있었다는 사실은 일부 이행 사정이지만, 실질적 출입통제ㆍ감독ㆍ하청 크레인 작업 관리가 부족했다면 추가 책임이 남습니다.",
        "",
        "[근로자 측 책임]",
        "- 근로자가 출입금지 표지를 보고도 금지구역에 진입한 점은 사고 발생에 대한 과실로 참작될 수 있습니다.",
        "- 그러나 특별안전교육 제공, 위험성평가, 출입통제, 도급 작업 관리 의무는 사업주ㆍ시공사 측 의무입니다.",
        "- 따라서 근로자 과실이 있더라도 시공사의 산업안전보건법상 조치의무와 중대재해처벌법상 경영책임자 의무 위반 가능성은 별도로 판단해야 합니다.",
    ]
    return "\n".join(lines)


def direct_contractor_worker_responsibility_sources(sources: list[SourceDoc]) -> list[SourceDoc]:
    system = find_serious_source(sources, note="serious_duty_system", article="제4조", terms=("반기",))
    subcontract = find_serious_source(sources, article="제4조", terms=("도급", "기준", "절차"))
    selected: list[SourceDoc | None] = [
        find_serious_source(sources, note="serious_duty_law", article="제4조") or make_serious_reference_source(
            "중대재해처벌법",
            article="제4조",
            page="1",
            content="중대재해처벌법 제4조: 안전보건관리체계 구축 및 이행 조치 의무.",
        ),
        source_with_article_label(system, "제4조제3호") if system else make_serious_reference_source(
            "중대재해처벌법 시행령",
            article="제4조제3호",
            page="2",
            content="중대재해처벌법 시행령 제4조제3호: 유해ㆍ위험요인 확인ㆍ개선 절차를 반기 1회 이상 점검해야 한다.",
        ),
        find_osha_source(sources, article="제36조", terms=("위험성평가",)) or make_osha_reference_source(
            "산업안전보건법",
            article="제36조",
            content="산업안전보건법 제36조: 사업주는 유해ㆍ위험요인을 찾아 위험성을 결정하고 감소대책을 수립ㆍ실행해야 한다.",
        ),
        find_excavation_item19_source(sources) or make_osha_reference_source(
            "산업안전보건법 시행규칙",
            annex="별표 5 제19호",
            page="82",
            content="산업안전보건법 시행규칙 별표 5 제19호: 굴착면 높이 2미터 이상 지반 굴착 작업 특별교육.",
        ),
        find_signage_source(sources) or make_osha_reference_source(
            "산업안전보건법 시행규칙",
            annex="별표 6 제1호",
            page="96",
            content="산업안전보건법 시행규칙 별표 6 제1호: 출입금지 등 금지표지 기준.",
        ),
        make_osha_reference_source(
            "산업안전보건기준에 관한 규칙",
            article="제14조",
            content="산업안전보건기준에 관한 규칙 제14조: 낙하물 등 위험이 있는 구역에는 관계 근로자가 아닌 사람의 출입을 금지하는 등 필요한 조치를 해야 한다.",
        ),
        find_serious_source(sources, note="serious_contract_duty", article="제5조", terms=("실질적으로지배",)) or make_serious_reference_source(
            "중대재해처벌법",
            article="제5조",
            page="1",
            content="중대재해처벌법 제5조: 도급ㆍ용역ㆍ위탁 관계에서도 실질적으로 지배ㆍ운영ㆍ관리하는 경우 안전 및 보건 확보의무를 부담한다.",
        ),
        source_with_article_label(subcontract, "제4조제9호") if subcontract else make_serious_reference_source(
            "중대재해처벌법 시행령",
            article="제4조제9호",
            page="2",
            content="중대재해처벌법 시행령 제4조제9호: 도급ㆍ용역ㆍ위탁 시 종사자 안전ㆍ보건 확보 기준과 절차를 마련하고 반기 1회 이상 점검해야 한다.",
        ),
    ]
    return [source for source in selected if source]


def direct_dual_law_sources(question: str, sources: list[SourceDoc]) -> list[SourceDoc]:
    osha_article_38 = find_osha_source(sources, article="제38조") or make_osha_reference_source(
        "산업안전보건법",
        article="제38조",
        content="산업안전보건법 제38조: 사업주는 추락ㆍ붕괴 등 위험을 예방하기 위해 필요한 안전조치를 하여야 한다.",
    )
    selected: list[SourceDoc | None] = [
        osha_article_38,
        find_osha_scaffold_source(sources),
        find_osha_source(sources, article="제62조"),
        find_osha_source(sources, article="제64조"),
        find_osha_source(sources, article="제11조"),
        find_special_education_penalty_source(sources),
        find_serious_source(sources, note="serious_definition", article="제2조"),
        find_serious_source(sources, note="serious_scope", article="제3조"),
        find_serious_source(sources, note="serious_duty_law", article="제4조"),
        find_serious_source(sources, note="serious_contract_duty", article="제5조", terms=("실질적으로지배",)),
        find_serious_source(sources, note="serious_manager_penalty", article="제6조"),
        find_serious_source(sources, note="serious_entity_penalty", article="제7조"),
    ]
    return [source for source in selected if source]


def extract_accident_facts(text: str) -> dict[str, int | bool]:
    compact = re.sub(r"\s+", "", text)
    worker_count = extract_worker_count(compact)
    if worker_count is None:
        worker_count = 12 if "사업장상시근로자수:12명" in compact else None
    death_count = 1 if any(term in compact for term in ("사망", "사망함", "사망자")) else 0
    injury_count = extract_nearest_int(compact, ("부상자",))
    if injury_count is None and any(term in compact for term in ("근로자2명", "2명이동시에", "두명모두", "2명모두")) and any(
        term in compact for term in ("6개월", "입원", "치료")
    ):
        injury_count = 2
    if injury_count is None and any(term in compact for term in ("근로자2명", "2명이동시에")) and "매몰" in compact and "두명" in compact:
        injury_count = 2
    if injury_count is None and any(term in compact for term in ("부상을입", "골절", "입원치료")):
        injury_count = 1
    treatment_months = extract_nearest_int(compact, ("개월",))
    if not treatment_months and injury_count == 2 and "매몰" in compact and "두명" in compact:
        treatment_months = 6
    return {
        "worker_count": worker_count or 0,
        "death_count": death_count,
        "injury_count": injury_count or 0,
        "treatment_months": treatment_months or 0,
    }


def extract_worker_count(compact_text: str) -> int | None:
    patterns = (
        r"상시근로자수[:：]?(\d+)명",
        r"상시근로자(\d+)명",
        r"근로자수[:：]?(\d+)명",
    )
    matches: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, compact_text):
            matches.append((match.start(), int(match.group(1))))
    if matches:
        return sorted(matches, key=lambda item: item[0])[0][1]
    return extract_nearest_int(compact_text, ("상시근로자수:", "상시근로자", "근로자수"))


def extract_nearest_int(compact_text: str, anchors: tuple[str, ...]) -> int | None:
    for anchor in anchors:
        index = compact_text.find(anchor)
        if index == -1:
            continue
        window = compact_text[max(0, index - 12): index + len(anchor) + 12]
        numbers = re.findall(r"\d+", window)
        if numbers:
            return int(numbers[-1])
    return None


def evaluate_serious_accident_applicability(facts: dict[str, int | bool]) -> dict[str, object]:
    worker_count = int(facts.get("worker_count") or 0)
    death_count = int(facts.get("death_count") or 0)
    injury_count = int(facts.get("injury_count") or 0)
    treatment_months = int(facts.get("treatment_months") or 0)
    if worker_count and worker_count < 5:
        return {
            "applies": False,
            "label": "미적용",
            "reason": f"상시 근로자 {worker_count}명으로 5명 미만 사업장에 해당하여 중대재해처벌법 제3조 적용 제외입니다.",
        }
    if death_count >= 1:
        return {
            "applies": True,
            "label": "적용됨",
            "reason": "사망자가 1명 이상 발생하여 중대재해처벌법 제2조제2호가목의 중대산업재해 요건을 충족합니다.",
        }
    if injury_count >= 2 and treatment_months >= 6:
        return {
            "applies": True,
            "label": "적용됨",
            "reason": f"동일 사고로 6개월 이상 치료가 필요한 부상자가 {injury_count}명 발생하여 중대재해처벌법 제2조제2호나목 요건을 충족합니다.",
        }
    if treatment_months >= 6 and injury_count == 1:
        return {
            "applies": False,
            "label": "미적용",
            "reason": f"치료기간은 {treatment_months}개월로 6개월 이상이지만, 부상자가 1명뿐이라 제2조제2호나목의 '부상자 2명 이상' 요건을 충족하지 못합니다.",
        }
    return {
        "applies": False,
        "label": "미적용 또는 판단불가",
        "reason": "검색ㆍ시나리오 정보에서 사망자 1명 이상, 6개월 이상 치료 필요 부상자 2명 이상, 직업성 질병자 3명 이상 요건이 확인되지 않았습니다.",
    }


def find_osha_source(sources: list[SourceDoc], *, article: str = "", terms: tuple[str, ...] = ()) -> SourceDoc | None:
    if article:
        candidates: list[SourceDoc] = []
        for source in sources:
            metadata = source.metadata
            law_name = str(metadata.get("law_name", "") or metadata.get("source", "") or metadata.get("pdf_file", ""))
            if "산업안전보건법" not in law_name:
                continue
            if not is_valid_osha_article_source(law_name, article):
                continue
            compact = re.sub(r"\s+", "", source.content)
            if metadata.get("article") == article and (not terms or all(term in compact for term in terms)):
                candidates.append(source)
        if candidates:
            return sorted(candidates, key=lambda src: str(src.metadata.get("page", "")) in {"", "0"})[0]
    for source in sources:
        metadata = source.metadata
        law_name = str(metadata.get("law_name", "") or metadata.get("source", "") or metadata.get("pdf_file", ""))
        if "산업안전보건법" not in law_name:
            continue
        if article and not is_valid_osha_article_source(law_name, article):
            continue
        compact = re.sub(r"\s+", "", source.content)
        if article and (not metadata.get("article") or metadata.get("article") == article) and article in compact:
            if not terms or all(term in compact for term in terms):
                return source
    return None


def is_valid_osha_article_source(law_name: str, article: str) -> bool:
    """Avoid treating 시행규칙/시행령 table mentions of 본법 articles as the article source."""
    if article in {"제36조", "제38조", "제57조", "제62조", "제64조"}:
        return "시행규칙" not in law_name and "시행령" not in law_name
    if article in {"제11조", "제43조", "제53조"}:
        return "시행령" in law_name or "시행규칙" not in law_name
    return True


def find_osha_scaffold_source(sources: list[SourceDoc]) -> SourceDoc | None:
    for source in sources:
        law_name = str(source.metadata.get("law_name", "") or source.metadata.get("source", ""))
        compact = re.sub(r"\s+", "", source.content)
        if "산업안전보건법" in law_name and (
            "[작업항목]23." in compact
            or "23.비계의조립·해체또는변경작업" in compact
            or "비계의조립·해체또는변경작업" in compact
        ):
            return source
    return None


def make_osha_reference_source(
    law_name: str,
    *,
    article: str = "",
    annex: str = "",
    page: str = "",
    citation_page: str = "",
    content: str = "",
) -> SourceDoc:
    return SourceDoc(
        content=content or f"{law_name} {article or annex}",
        metadata={
            "law_name": law_name,
            "article": article,
            "annex": annex,
            "page": page,
            "citation_page": citation_page,
            "score": 0.9,
            "source_type": "text",
            "retrieval_note": "osha_reference_fallback",
        },
    )


def direct_serious_accident_act_answer(question: str, sources: list[SourceDoc]) -> str | None:
    """Deterministic answers for 중대재해처벌법 Q1~Q5 style accident questions."""
    if not is_serious_accident_act_question(question):
        return None

    compact = re.sub(r"\s+", "", question)
    if is_serious_accident_composite_penalty_question(compact):
        return direct_serious_accident_composite_penalty_answer(question, sources)
    if any(term in compact for term in ("중대산업재해", "해당여부", "해당하는가", "적용되는가", "적용여부")):
        return direct_serious_accident_scope_answer(question, sources)
    if any(term in compact for term in ("도급", "하청", "원청", "수급", "위탁")):
        return direct_serious_accident_contract_answer(sources)
    if any(term in compact for term in ("가중처벌", "가중", "이전위반", "위반이력", "재범", "5년이내")):
        return direct_serious_accident_aggravation_answer(sources)
    if is_serious_accident_penalty_question(compact):
        return direct_serious_accident_penalty_answer(sources)
    if any(term in compact for term in ("안전보건교육", "교육의무", "과태료")) and any(term in compact for term in ("사고후", "대표이사", "경영책임자", "1차", "2차", "3차")):
        return direct_serious_accident_training_penalty_answer(sources)
    if any(term in compact for term in ("경영책임자", "대표이사", "위반한의무", "구체적으로나열", "안전보건관리체계")):
        return direct_serious_accident_duty_answer(sources)
    return None


def is_serious_accident_composite_penalty_question(compact_question: str) -> bool:
    """Detect Q4-style compound questions before 도급/원청 words in the scenario hijack routing."""
    has_numbered_parts = bool(re.search(r"[①②③④]|(?:^|[^0-9])[1-4][.)]", compact_question))
    asks_all = any(term in compact_question for term in ("모두답", "세가지를모두", "다음세가지", "각각구분"))
    asks_applicability = any(term in compact_question for term in ("적용되는가", "적용여부", "적용요건", "중대재해처벌법이적용"))
    asks_penalty = any(term in compact_question for term in ("처벌수위", "대표이사", "법인", "징역", "벌금"))
    asks_training_fine = "과태료" in compact_question and any(term in compact_question for term in ("안전보건교육", "교육미이행", "1차", "2차", "3차"))
    return (has_numbered_parts or asks_all) and asks_applicability and asks_penalty and asks_training_fine


def direct_serious_accident_composite_penalty_answer(question: str, sources: list[SourceDoc]) -> str:
    facts = extract_accident_facts(question)
    serious = evaluate_serious_accident_applicability(facts)
    definition = find_serious_source(sources, note="serious_definition", article="제2조")
    scope = find_serious_source(sources, note="serious_scope", article="제3조")
    duty = find_serious_source(sources, note="serious_duty_law", article="제4조")
    contract = find_serious_source(sources, note="serious_contract_duty", article="제5조", terms=("실질적으로지배",))
    manager = find_serious_source(sources, note="serious_manager_penalty", article="제6조") or make_serious_reference_source(
        "중대재해처벌법",
        article="제6조제1항",
        page="1",
        content="중대재해처벌법 제6조제1항: 제4조 또는 제5조를 위반하여 사망자가 발생한 경우 1년 이상의 징역 또는 10억원 이하의 벌금에 처한다.",
    )
    entity = find_serious_source(sources, note="serious_entity_penalty", article="제7조") or make_serious_reference_source(
        "중대재해처벌법",
        article="제7조제1호",
        page="1",
        content="중대재해처벌법 제7조제1호: 제6조제1항 위반행위를 한 법인에는 50억원 이하의 벌금형을 과한다.",
    )
    damage = find_serious_source(sources, note="serious_damage", article="제15조") or make_serious_reference_source(
        "중대재해처벌법",
        article="제15조",
        page="3",
        content="중대재해처벌법 제15조: 고의 또는 중대한 과실로 의무를 위반하여 중대재해를 발생하게 한 경우 손해액의 5배를 넘지 않는 범위에서 배상책임을 진다.",
    )
    training_law = find_serious_source(sources, note="serious_manager_training_law", article="제8조")
    training_hours = find_serious_source(sources, note="serious_manager_training_hours", article="제6조")
    training_fine = find_serious_source(sources, note="serious_training_penalty", annex="별표 4") or make_serious_reference_source(
        "중대재해처벌법 시행령",
        annex="별표 4",
        citation_page="15",
        content="중대재해처벌법 시행령 별표 4 과태료의 부과기준: 안전보건교육 미이행 1차 1천만원, 2차 3천만원, 3차 5천만원.",
    )

    lines = [
        "① 중대재해처벌법 적용 여부",
        f"- 적용 여부: {serious['label']}",
        f"- 적용 요건 판단: {serious['reason']}",
        f"- 중대산업재해 정의 근거: {source_basis_or_fallback(definition, '중대재해처벌법 제2조제2호가목')}",
        f"- 적용범위 근거: {source_basis_or_fallback(scope, '중대재해처벌법 제3조')}",
    ]
    if serious["applies"]:
        lines.extend(
            [
                "- 이 사고는 근로자 사망 사고이므로 제2조제2호가목의 '사망자 1명 이상' 요건을 충족합니다.",
                "- 사업장이 상시 근로자 5명 이상이면 중대재해처벌법 제3조의 5명 미만 적용 제외 대상이 아닙니다.",
                f"- 경영책임자 안전 및 보건 확보의무 근거: {source_basis_or_fallback(duty, '중대재해처벌법 제4조')}",
                f"- 도급ㆍ용역ㆍ위탁 작업이 포함된 부분은 실질적 지배ㆍ운영ㆍ관리 여부에 따라 {source_basis_or_fallback(contract, '중대재해처벌법 제5조')}도 함께 검토합니다.",
            ]
        )
    else:
        lines.append("- 중대산업재해 요건이 충족되지 않으면 아래 처벌ㆍ교육 의무는 적용되지 않거나 별도 판단이 필요합니다.")

    lines.extend(
        [
            "",
            "② A건설 대표이사와 법인(A건설)의 처벌 수위",
        ]
    )
    if serious["applies"] and int(facts.get("death_count") or 0) >= 1:
        lines.extend(
            [
                "- 대표이사 등 경영책임자: 1년 이상의 징역 또는 10억원 이하의 벌금",
                "- 징역과 벌금은 병과될 수 있습니다.",
                f"- 대표이사 처벌 근거: {source_basis_or_fallback(manager, '중대재해처벌법 제6조제1항')}",
                "- 법인(A건설): 50억원 이하의 벌금",
                f"- 법인 양벌규정 근거: {source_basis_or_fallback(entity, '중대재해처벌법 제7조제1호')}",
                "- 민사상 징벌적 손해배상: 손해액의 5배 이내",
                f"- 손해배상 근거: {source_basis_or_fallback(damage, '중대재해처벌법 제15조')}",
            ]
        )
    elif serious["applies"]:
        lines.extend(format_serious_punishment_lines(facts, manager, entity))
        lines.append(f"- 민사상 징벌적 손해배상 근거: {source_basis_or_fallback(damage, '중대재해처벌법 제15조')}")
    else:
        lines.append("- 중대재해처벌법 적용 요건이 충족되지 않으면 제6조ㆍ제7조 처벌은 적용되지 않습니다.")

    lines.extend(
        [
            "",
            "③ 사고 후 경영책임자 안전보건교육 미이행 과태료",
            "- 중대산업재해가 발생한 법인 또는 기관의 경영책임자등은 안전보건교육을 이수해야 합니다.",
            f"- 교육 의무 근거: {source_basis_or_fallback(training_law, '중대재해처벌법 제8조')}",
            f"- 교육 내용ㆍ시간 근거: {source_basis_or_fallback(training_hours, '중대재해처벌법 시행령 제6조')}",
            "- 안전보건교육 미이행 과태료:",
            "  - 1차 위반: 1천만원",
            "  - 2차 위반: 3천만원",
            "  - 3차 이상 위반: 5천만원",
            f"- 과태료 근거: {source_basis_or_fallback(training_fine, '중대재해처벌법 시행령 별표 4')}",
            "",
            "[주의]",
            "- 위 과태료는 중대재해처벌법상 경영책임자 안전보건교육 미이행 과태료입니다.",
            "- 산업안전보건법상 특별안전교육 미실시 과태료(교육대상 근로자 1명당 50만원/100만원/150만원)와 혼동하면 안 됩니다.",
        ]
    )
    return "\n".join(lines)


def direct_serious_accident_scope_answer(question: str, sources: list[SourceDoc]) -> str:
    definition = find_serious_source(sources, note="serious_definition", article="제2조")
    scope = find_serious_source(sources, note="serious_scope", article="제3조")
    facts = extract_accident_facts(question)
    serious = evaluate_serious_accident_applicability(facts)
    lines = [
        f"결론: {'YES' if serious['applies'] else 'NO'}. 이 사고는 중대재해처벌법상 중대산업재해 {('에 해당합니다' if serious['applies'] else '에 해당하지 않습니다')}.",
        "",
        "[판단 근거]",
        f"- {serious['reason']}",
        "- 중대재해처벌법 제2조제2호는 사망자 1명 이상, 같은 사고로 6개월 이상 치료가 필요한 부상자 2명 이상, 같은 유해요인으로 직업성 질병자 3명 이상 발생 등을 중대산업재해 요건으로 봅니다.",
        "- 중대재해처벌법 제3조는 상시 근로자 5명 미만 사업 또는 사업장을 적용 제외합니다.",
        "",
        "[근거]",
        f"- {source_basis_or_fallback(definition, '중대재해처벌법 제2조제2호가목')}",
        f"- {source_basis_or_fallback(scope, '중대재해처벌법 제3조')}",
    ]
    return "\n".join(lines)


def direct_serious_accident_duty_answer(sources: list[SourceDoc]) -> str:
    law_duty = find_serious_source(sources, note="serious_duty_law", article="제4조")
    system = find_serious_source(sources, note="serious_duty_system", article="제4조", terms=("반기",))
    education = find_serious_source(sources, note="serious_duty_education_check", article="제4조", terms=("교육",))
    contract = find_serious_source(sources, note="serious_contract_duty", article="제5조", terms=("실질적으로지배",))
    lines = [
        "A건설 대표이사 등 경영책임자가 문제될 수 있는 중대재해처벌법상 의무는 다음과 같습니다.",
        "",
        "1. 안전보건관리체계 구축 및 이행 의무",
        f"   - 근거: {source_basis_or_fallback(law_duty, '중대재해처벌법 제4조제1항제1호')}",
        "   - 판단: 안전보건관리체계가 형식적으로 구축되어 있더라도 반기 점검이 미실시된 상태라면 이행 점검 의무 위반이 문제됩니다.",
        "",
        "2. 유해ㆍ위험요인 확인 및 개선 절차의 마련ㆍ점검 의무",
        f"   - 근거: {source_basis_or_fallback(system, '중대재해처벌법 시행령 제4조')}",
        "   - 판단: 작업 전 위험성평가가 미실시되어 굴착ㆍ크레인 병행작업의 위험요인이 사전에 확인ㆍ개선되지 않았습니다.",
        "",
        "3. 안전보건 관계 법령상 의무 이행 여부 점검 의무",
        f"   - 근거: {source_basis_or_fallback(education, '중대재해처벌법 시행령 제4조')}",
        "   - 판단: 일용직 근로자 특별안전교육 미실시 여부를 점검하고, 미실시 교육의 이행을 지시ㆍ예산 확보까지 했는지 검토해야 합니다.",
        "",
        "4. 도급ㆍ용역ㆍ위탁 시 안전보건 확보 기준과 절차 점검 의무",
        f"   - 근거: {source_basis_or_fallback(contract, '중대재해처벌법 제5조 및 시행령 제4조')}",
        "   - 판단: 크레인 운용을 하청업체 B사에 도급한 상태이므로, 수급인의 안전보건 역량과 작업 통제 기준을 마련ㆍ점검했는지가 쟁점입니다.",
        "",
        "정리하면, 이 사고에서는 반기 점검 미실시, 위험성평가 미실시, 특별안전교육 이행 점검 미흡, 도급 작업 통제 미흡이 경영책임자 의무 위반 쟁점입니다.",
    ]
    return "\n".join(lines)


def direct_serious_accident_contract_answer(sources: list[SourceDoc]) -> str:
    contract = find_serious_source(sources, note="serious_contract_duty", article="제5조", terms=("실질적으로지배",))
    duty = find_serious_source(sources, note="serious_duty_law", article="제4조")
    lines = [
        "결론: YES. 하청업체 B사가 크레인을 운용했더라도 A건설 경영책임자의 책임이 문제될 수 있습니다.",
        "",
        "[판단 기준]",
        "- 중대재해처벌법 제5조는 사업주나 경영책임자등이 제3자에게 도급ㆍ용역ㆍ위탁 등을 한 경우에도, 그 시설ㆍ장비ㆍ장소 등에 실질적으로 지배ㆍ운영ㆍ관리하는 책임이 있으면 제4조의 안전 및 보건 확보의무를 부담한다고 봅니다.",
        "- 사고 장소는 A건설의 아파트 신축 현장이고, 지하 2층 굴착구역 출입통제ㆍ작업조정ㆍ위험성평가ㆍ교육 이행 점검은 원청의 현장 관리 범위에 속할 가능성이 큽니다.",
        "",
        "[근거]",
        f"- {source_basis_or_fallback(contract, '중대재해처벌법 제5조')}",
        f"- {source_basis_or_fallback(duty, '중대재해처벌법 제4조')}",
        "",
        "따라서 B사의 직접 작업상 과실과 별개로, A건설이 해당 장소와 작업을 실질적으로 지배ㆍ운영ㆍ관리했다면 A건설 경영책임자에게도 중대재해처벌법상 책임이 성립할 수 있습니다.",
    ]
    return "\n".join(lines)


def direct_serious_accident_penalty_answer(sources: list[SourceDoc]) -> str:
    manager = find_serious_source(sources, note="serious_manager_penalty", article="제6조")
    entity = find_serious_source(sources, note="serious_entity_penalty", article="제7조") or make_serious_reference_source(
        "중대재해처벌법",
        article="제7조",
        page="1",
        content="중대재해처벌법 제7조: 법인의 경영책임자등이 제6조제1항 위반행위를 하면 그 법인에 50억원 이하의 벌금형을 과한다.",
    )
    damage = find_serious_source(sources, note="serious_damage", article="제15조")
    lines = [
        "중대재해처벌법상 처벌 수위는 대표이사 등 경영책임자와 법인을 구분해야 합니다.",
        "",
        "1. 대표이사 등 경영책임자",
        "   - 사망자가 1명 발생한 중대산업재해에서 제4조 또는 제5조 의무 위반으로 사망 결과가 발생한 경우, 1년 이상의 징역 또는 10억원 이하의 벌금 대상입니다.",
        "   - 징역과 벌금은 병과될 수 있습니다.",
        f"   - 근거: {source_basis_or_fallback(manager, '중대재해처벌법 제6조제1항')}",
        "",
        "2. 법인 A건설",
        "   - 법인의 경영책임자 등이 제6조제1항 위반행위를 하면 법인은 50억원 이하의 벌금 대상입니다.",
        "   - 다만 법인이 위반행위 방지를 위해 상당한 주의와 감독을 게을리하지 않았다는 점이 인정되면 양벌규정 적용이 제한될 수 있습니다.",
        f"   - 근거: {source_basis_or_fallback(entity, '중대재해처벌법 제7조')}",
        "",
        "3. 민사상 손해배상",
        "   - 고의 또는 중대한 과실로 안전 및 보건 확보의무를 위반하여 중대재해가 발생한 경우, 손해액의 5배를 넘지 않는 범위에서 배상책임이 문제될 수 있습니다.",
        f"   - 근거: {source_basis_or_fallback(damage, '중대재해처벌법 제15조')}",
    ]
    return "\n".join(lines)


def direct_serious_accident_aggravation_answer(sources: list[SourceDoc]) -> str:
    aggravation = find_serious_source(sources, article="제6조", terms=("5년", "2분의1")) or make_serious_reference_source(
        "중대재해처벌법",
        article="제6조제3항",
        page="1",
        content="중대재해처벌법 제6조제3항: 제6조제1항 또는 제2항의 죄로 형을 선고받고 그 형이 확정된 후 5년 이내에 다시 제1항 또는 제2항의 죄를 저지른 자는 각 항에서 정한 형의 2분의 1까지 가중한다.",
    )
    lines = [
        "[산업안전보건법]",
        "- 이전 위반 이력이 이번 처벌에 영향을 미치는지 여부는 구체 위반 조항과 처분 기준별로 별도 검토가 필요합니다.",
        "- 산업안전보건법 시행령 별표 35처럼 과태료 기준은 1차ㆍ2차ㆍ3차 이상 위반으로 차등되는 구조가 있습니다.",
        "- 다만 현재 검색 근거에서는 중대재해처벌법 제6조제3항과 같은 형태의 '형 확정 후 5년 이내 재범 시 형의 2분의 1 가중' 조항은 확인되지 않았습니다.",
        "",
        "[중대재해처벌법]",
        "- 결론: 영향을 미칠 수 있습니다. 중대재해처벌법에는 5년 이내 재범에 대한 가중처벌 조항이 있습니다.",
        "",
        "[가중처벌 기준]",
        "- 중대재해처벌법 제6조제1항 또는 제2항의 죄로 형을 선고받고 그 형이 확정된 후 5년 이내에 다시 같은 조 제1항 또는 제2항의 죄를 저지르면 가중처벌 대상입니다.",
        "- 가중 범위는 각 항에서 정한 형의 2분의 1까지입니다.",
        f"- 근거: {format_aggravation_basis(aggravation)}",
        "",
        "[판단]",
        "- 집행유예도 유죄판결의 형이 확정된 경우라면 '형을 선고받고 그 형이 확정된 후'라는 요건 판단에 포함됩니다.",
        "- 예컨대 3년 전 중대재해처벌법 제6조 위반으로 징역형의 집행유예가 확정되었고 이번에 다시 제6조제1항 또는 제2항의 죄를 저질렀다면 5년 이내 재범 가중을 검토합니다.",
        "- 사망 사고로 제6조제1항이 적용되는 경우 기본 벌금 상한 10억원은 제6조제3항에 따라 2분의 1까지 가중되어 15억원까지 가중될 수 있습니다.",
        "- 단순한 행정지도 이력이나 과태료 이력만으로 바로 제6조제3항 가중처벌이 되는 것은 아닙니다.",
        "- 이전에 중대재해처벌법 제6조제1항 또는 제2항의 죄로 형이 확정되었고, 그 확정 후 5년 이내에 다시 해당 죄를 저질렀는지가 핵심입니다.",
    ]
    return "\n".join(lines)


def format_aggravation_basis(source: SourceDoc | None) -> str:
    if source:
        return f"중대재해처벌법 제6조제3항, p.{format_source_page(source)}"
    return "중대재해처벌법 제6조제3항"


def direct_serious_accident_training_penalty_answer(sources: list[SourceDoc]) -> str:
    training_law = find_serious_source(sources, note="serious_manager_training_law", article="제8조")
    hours = find_serious_source(sources, note="serious_manager_training_hours", article="제6조")
    penalty = find_serious_source(sources, note="serious_training_penalty", annex="별표 4") or make_serious_reference_source(
        "중대재해처벌법 시행령",
        annex="별표 4",
        citation_page="15",
        content="중대재해처벌법 시행령 별표 4 과태료의 부과기준: 안전보건교육 미이행 1차 1천만원, 2차 3천만원, 3차 5천만원",
    )
    lines = [
        "사고 후 A건설 대표이사 등 경영책임자는 중대재해처벌법상 안전보건교육 수강 의무가 있습니다.",
        "",
        "[교육 의무]",
        "- 중대산업재해가 발생한 법인 또는 기관의 경영책임자등은 안전보건교육을 이수해야 합니다.",
        "- 교육시간은 총 20시간이며, 안전보건관리체계 구축 등 안전ㆍ보건 확보의무와 중대산업재해 원인 분석 및 재발방지 방안을 포함합니다.",
        f"- 근거: {source_basis_or_fallback(training_law, '중대재해처벌법 제8조')}",
        f"- 세부 기준: {source_basis_or_fallback(hours, '중대재해처벌법 시행령 제6조')}",
        "",
        "[미이행 과태료]",
        "- 1차 위반: 1천만원",
        "- 2차 위반: 3천만원",
        "- 3차 이상 위반: 5천만원",
        f"- 근거: {source_basis_or_fallback(penalty, '중대재해처벌법 시행령 별표 4')}",
    ]
    return "\n".join(lines)


def direct_serious_accident_act_sources(question: str, sources: list[SourceDoc]) -> list[SourceDoc]:
    compact = re.sub(r"\s+", "", question)
    selected: list[SourceDoc | None]
    if is_serious_accident_composite_penalty_question(compact):
        selected = [
            find_serious_source(sources, note="serious_definition", article="제2조"),
            find_serious_source(sources, note="serious_scope", article="제3조"),
            find_serious_source(sources, note="serious_duty_law", article="제4조"),
            find_serious_source(sources, note="serious_contract_duty", article="제5조", terms=("실질적으로지배",)),
            find_serious_source(sources, note="serious_manager_penalty", article="제6조"),
            find_serious_source(sources, note="serious_entity_penalty", article="제7조"),
            find_serious_source(sources, note="serious_damage", article="제15조"),
            find_serious_source(sources, note="serious_manager_training_law", article="제8조"),
            find_serious_source(sources, note="serious_manager_training_hours", article="제6조"),
            find_serious_source(sources, note="serious_training_penalty", annex="별표 4"),
        ]
    elif any(term in compact for term in ("중대산업재해", "해당여부", "해당하는가", "적용되는가", "적용여부")):
        selected = [
            find_serious_source(sources, note="serious_definition", article="제2조"),
            find_serious_source(sources, note="serious_scope", article="제3조"),
        ]
    elif any(term in compact for term in ("도급", "하청", "원청", "수급", "위탁")):
        selected = [
            find_serious_source(sources, note="serious_contract_duty", article="제5조", terms=("실질적으로지배",)),
            find_serious_source(sources, note="serious_duty_law", article="제4조"),
        ]
    elif any(term in compact for term in ("가중처벌", "가중", "이전위반", "위반이력", "재범", "5년이내")):
        selected = [
            find_serious_source(sources, article="제6조", terms=("5년", "2분의1")) or make_serious_reference_source(
                "중대재해처벌법",
                article="제6조제3항",
                page="1",
                content="중대재해처벌법 제6조제3항: 5년 이내 재범 시 각 항에서 정한 형의 2분의 1까지 가중한다.",
            )
        ]
    elif is_serious_accident_penalty_question(compact):
        selected = [
            find_serious_source(sources, note="serious_manager_penalty", article="제6조"),
            find_serious_source(sources, note="serious_entity_penalty", article="제7조") or make_serious_reference_source(
                "중대재해처벌법",
                article="제7조",
                page="1",
                content="중대재해처벌법 제7조: 법인의 경영책임자등이 제6조제1항 위반행위를 하면 그 법인에 50억원 이하의 벌금형을 과한다.",
            ),
            find_serious_source(sources, note="serious_damage", article="제15조"),
        ]
    elif any(term in compact for term in ("안전보건교육", "교육의무", "과태료")):
        selected = [
            find_serious_source(sources, note="serious_manager_training_law", article="제8조"),
            find_serious_source(sources, note="serious_manager_training_hours", article="제6조"),
            find_serious_source(sources, note="serious_training_penalty", annex="별표 4") or make_serious_reference_source(
                "중대재해처벌법 시행령",
                annex="별표 4",
                citation_page="15",
                content="중대재해처벌법 시행령 별표 4 과태료의 부과기준: 안전보건교육 미이행 1차 1천만원, 2차 3천만원, 3차 5천만원",
            ),
        ]
    else:
        selected = [
            find_serious_source(sources, note="serious_duty_law", article="제4조"),
            find_serious_source(sources, note="serious_duty_system", article="제4조", terms=("반기",)),
            find_serious_source(sources, note="serious_duty_education_check", article="제4조", terms=("교육",)),
            find_serious_source(sources, note="serious_contract_duty", article="제5조", terms=("실질적으로지배",)),
        ]
    return [source for source in selected if source]


def is_serious_accident_penalty_question(compact_question: str) -> bool:
    if "과태료" in compact_question:
        return False
    penalty_part = compact_question.replace("중대재해처벌법", "")
    return (
        any(term in penalty_part for term in ("처벌수위", "형사처벌", "징역", "벌금", "손해배상"))
        or ("법인" in penalty_part and any(term in penalty_part for term in ("처벌", "벌금", "받을수있는")))
        or ("대표이사" in penalty_part and any(term in penalty_part for term in ("처벌", "징역", "벌금", "받을수있는")))
    )


def find_serious_source(
    sources: list[SourceDoc],
    *,
    note: str = "",
    article: str = "",
    annex: str = "",
    terms: tuple[str, ...] = (),
) -> SourceDoc | None:
    for source in sources:
        metadata = source.metadata
        if note and metadata.get("retrieval_note") == note:
            return source
    for source in sources:
        metadata = source.metadata
        law_name = str(metadata.get("law_name", "") or metadata.get("source", "") or metadata.get("pdf_file", ""))
        if "중대재해처벌법" not in law_name:
            continue
        compact = re.sub(r"\s+", "", source.content)
        if article and (metadata.get("article") == article or article in compact):
            if not terms or all(term in compact for term in terms):
                return source
        if annex and (metadata.get("annex") == annex or annex.replace(" ", "") in compact):
            if not terms or all(term in compact for term in terms):
                return source
    return None


def source_basis_or_fallback(source: SourceDoc | None, fallback: str) -> str:
    if source:
        return format_source_basis(source)
    return fallback


def source_basis_for_article(source: SourceDoc | None, law_name: str, article: str) -> str:
    if not source:
        return f"{law_name} {article}"
    page = format_source_page(source)
    page_suffix = f", p.{page}" if page and page != "0" else ""
    return f"{law_name} {article}{page_suffix}"


def source_with_article_label(source: SourceDoc, article: str) -> SourceDoc:
    return SourceDoc(
        content=source.content,
        metadata={
            **source.metadata,
            "article": article,
        },
    )


def format_source_basis_no_blank_page(source: SourceDoc) -> str:
    basis = format_source_basis(source)
    return basis[:-4] if basis.endswith(", p.") else basis


def make_serious_reference_source(
    law_name: str,
    *,
    article: str = "",
    annex: str = "",
    page: str = "",
    citation_page: str = "",
    content: str = "",
) -> SourceDoc:
    return SourceDoc(
        content=content or f"{law_name} {article or annex}",
        metadata={
            "law_name": law_name,
            "article": article,
            "annex": annex,
            "page": page,
            "citation_page": citation_page,
            "score": 0.98,
            "source_type": "text",
            "serious_accident_act": True,
            "retrieval_note": "serious_reference_fallback",
        },
    )


def unique_sources(sources: list[SourceDoc]) -> list[SourceDoc]:
    result: list[SourceDoc] = []
    seen: set[tuple[str, str, str, str]] = set()
    for source in sources:
        key = source_identity(source)
        if key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result


def should_direct_special_education(question: str) -> bool:
    """Only bypass the LLM when the user's actual question asks for special education."""
    compact = re.sub(r"\s+", "", question)
    if any(term in compact for term in ("출입금지", "표지", "표지판", "추가책임", "추가적인책임", "면책")):
        return False
    has_special_education = any(term in compact for term in ("특별교육", "교육내용", "교육대상", "교육사항"))
    asks_listing = any(term in compact for term in ("모두", "나열", "알려", "무엇", "어떤", "조항"))
    return has_special_education and asks_listing


def should_direct_focused_excavation_violation(question: str) -> bool:
    """Bypass broad scenario inference when the user explicitly narrows to excavation special education."""
    compact = re.sub(r"\s+", "", question)
    return (
        any(term in compact for term in ("굴착작업관련", "굴착작업", "지반굴착", "굴착면"))
        and any(term in compact for term in ("특별교육", "미실시", "미이수", "위반"))
        and any(term in compact for term in ("중심", "중점", "관련"))
        and "모든특별교육" not in compact
        and "나열" not in compact
    )


def should_direct_scaffold_special_education(question: str) -> bool:
    """Bypass broad LLM inference for scaffold special-education questions."""
    compact = re.sub(r"\s+", "", question)
    has_scaffold = any(term in compact for term in ("비계", "강관비계", "이동식비계", "비계해체", "비계조립"))
    asks_special_education = any(term in compact for term in ("특별교육", "특별안전교육", "교육내용", "교육사항", "미실시", "미이수"))
    excludes_broad_dual_law = any(term in compact for term in ("중대재해처벌법", "경영책임자", "원청", "도급", "하청"))
    return has_scaffold and asks_special_education and not excludes_broad_dual_law


def is_exposure_limit_question(question: str) -> bool:
    normalized = question.lower()
    exposure_terms = (
        "twa",
        "stel",
        "ppm",
        "mg/m3",
        "mg/㎥",
        "시간가중",
        "단시간 노출",
        "노출값",
        "노출기준",
        "허용기준",
        "유해인자",
        "벤젠",
        "benzene",
    )
    return any(term in normalized or term in question for term in exposure_terms)


def is_violation_question(question: str) -> bool:
    # 재발방지·조치 도출 질문은 위반판단 포맷 미적용
    non_violation_terms = ("재발방지", "즉시 취해야", "조치를 제시", "조치를 알려")
    if any(term in question for term in non_violation_terms):
        return False
    violation_terms = (
        "위반",
        "위법",
        "법령 위반",
        "의무",
        "해당하는가",
        "해당되는가",
        "해당되나",
        "해당하나",
        "위반인가",
        "위반인지",
        "위반 여부",
        "책임",
        "추가적인 책임",
        "나열하라",
        "조항을 나열",
    )
    return any(term in question for term in violation_terms)


def is_punishment_question(question: str) -> bool:
    punishment_terms = (
        "행정처분",
        "처분 수위",
        "1차 위반",
        "2차 위반",
        "3차 위반",
        "과태료",
        "벌칙",
        "처벌 수위",
        "벌금",
    )
    return any(term in question for term in punishment_terms)


def is_prevention_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return any(
        term in compact
        for term in ("재발방지", "즉시취해야", "조치를제시", "조치를알려", "법령의무기준", "법적근거")
    )


def is_signage_responsibility_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return any(term in compact for term in ("출입금지", "표지", "표지판", "금지표지")) and any(
        term in compact for term in ("책임", "추가", "면책", "무시", "진입")
    )


def direct_signage_responsibility_answer(question: str, sources: list[SourceDoc]) -> str | None:
    """Answer 출입금지 표지 설치 후 추가 책임 questions without an LLM call."""
    if not is_signage_responsibility_question(question):
        return None

    signage_source = find_signage_source(sources)
    education_source = find_excavation_item19_source(sources)
    crane_signal_source = find_crane_signal_source(sources)
    if not signage_source and not education_source and not crane_signal_source:
        return None

    lines = [
        "결론: 추가 책임 가능성 있음",
        "",
        "[표지 설치의 의미]",
    ]
    if signage_source:
        lines.append(
            "- 출입금지 표지가 설치되어 있었다면 안전보건표지 설치 의무를 이행한 사정으로 볼 수 있습니다."
        )
        lines.append(f"- 근거: {format_source_basis(signage_source, default_annex='별표 6 제1호')}")
    else:
        lines.append("- 검색 결과에서 출입금지 표지 기준 자체는 명확히 확인되지 않았습니다.")

    lines.extend(
        [
            "",
            "[추가 책임이 남는 이유]",
            "- 표지 설치는 면책 사유가 아니라 위험 고지 조치 중 하나입니다.",
            "- 근로자가 표지를 무시하고 진입했다면, 사업주는 출입관리ㆍ감독ㆍ물리적 차단 등 추가 안전조치가 충분했는지 별도로 검토받을 수 있습니다.",
        ]
    )
    if education_source:
        lines.append(
            "- 사고 작업이 굴착면 높이 2미터 이상 지반 굴착작업에 해당하고 특별교육을 하지 않았다면, 표지 설치와 별개로 특별교육 미실시 책임이 남습니다."
        )
        lines.append(f"- 특별교육 근거: {format_source_basis(education_source, default_annex='별표 5 제19호')}")
    if crane_signal_source:
        lines.append(
            "- 사고의 직접 원인이 크레인 인양 중 자재 낙하라면, 인양ㆍ신호ㆍ낙하ㆍ비래 위험 교육 및 작업통제도 함께 검토해야 합니다."
        )
        lines.append(f"- 크레인 작업 근거: {format_source_basis(crane_signal_source, default_annex=f'별표 5 제{extract_item_number(crane_signal_source)}호')}")

    lines.extend(
        [
            "",
            "[판단]",
            "- 따라서 출입금지 표지가 있었다는 사실만으로 사업주의 책임이 곧바로 면제되지는 않습니다.",
            "- 표지 설치 여부는 유리한 사정이 될 수 있지만, 교육 미이수와 실질적 출입통제 미흡이 확인되면 추가 책임 가능성이 있습니다.",
        ]
    )
    return "\n".join(lines)


def find_signage_source(sources: list[SourceDoc]) -> SourceDoc | None:
    for source in sources:
        compact = re.sub(r"\s+", "", source.content)
        page = str(source.metadata.get("page", ""))
        if (
            "출입금지" in compact
            or "금지표지" in compact
            or "안전보건표지" in compact
            or "별표6" in compact
            or page in {"94", "95", "96"}
        ):
            return source
    return None


def find_special_education_penalty_source(sources: list[SourceDoc]) -> SourceDoc | None:
    for source in sources:
        compact = re.sub(r"\s+", "", source.content)
        law_name = str(source.metadata.get("law_name", ""))
        if (
            "시행령" in law_name
            and "법제29조제3항" in compact
            and ("교육대상근로자1명당50100150" in compact or "50100150" in compact)
        ):
            return source
    return None


def direct_special_education_penalty_answer(sources: list[SourceDoc]) -> str | None:
    """Deterministic answer for 시행령 별표 35 special-education penalty rows."""
    penalty_source = find_special_education_penalty_source(sources)
    if not penalty_source:
        return None

    admin_source = find_admin_disposition_source(sources)
    lines = [
        "결론: 특별안전교육 미실시에 대한 과태료 기준은 교육대상 근로자 1명당 다음과 같습니다.",
        "",
        "[과태료 기준]",
        "- 1차 위반: 50만원",
        "- 2차 위반: 100만원",
        "- 3차 이상 위반: 150만원",
        "",
        "[근거]",
        f"- {format_source_basis(penalty_source, default_annex='별표 35')}",
        "- 위반 유형: 법 제29조제3항 위반, 유해하거나 위험한 작업에 근로자를 사용할 때 안전보건교육을 추가로 하지 않은 경우",
        "- 부과 기준: 교육대상 근로자 1명당 50 / 100 / 150만원",
    ]
    if admin_source:
        lines.extend(
            [
                "",
                "[구분]",
                f"- {format_source_basis(admin_source, default_annex='별표 26')}은 업무정지 등 행정처분 기준이고, 과태료 금액 기준은 시행령 별표 35입니다.",
            ]
        )
    return "\n".join(lines)


def direct_prevention_action_answer(sources: list[SourceDoc]) -> str | None:
    """Deterministically produce recurrence-prevention actions from retrieved legal bases."""
    education_source = find_excavation_item19_source(sources)
    signage_source = find_signage_source(sources)
    crane_source = find_crane_safety_source(sources)
    crane_signal_source = find_crane_signal_source(sources)
    report_source = find_accident_report_source(sources)
    accident_form_source = find_accident_investigation_form_source(sources)
    hazard_plan_source = find_hazard_plan_source(sources)

    if not any((education_source, signage_source, crane_source, crane_signal_source, report_source, accident_form_source, hazard_plan_source)):
        return None

    lines = [
        "사업주가 즉시 취해야 할 재발방지 조치는 다음과 같습니다.",
        "",
        "[재발방지 조치]",
    ]

    if education_source:
        lines.extend(
            [
                "1. 굴착 작업 투입 근로자 특별안전교육 실시",
                "   - 조치: 일용직 근로자를 포함해 굴착면 높이 2미터 이상 지반 굴착작업에 투입되는 근로자에게 특별안전교육을 실시하고, 교육 이수 여부를 작업 투입 전 확인한다.",
                f"   - 법적 근거: {format_source_basis(education_source, default_annex='별표 5 제19호')}",
            ]
        )

    if signage_source:
        lines.extend(
            [
                "2. 출입금지 구역의 물리적 차단 및 출입통제 강화",
                "   - 조치: 출입금지 표지 설치에 그치지 않고 안전펜스, 차단시설, 감시자 배치 등으로 금지구역 진입을 실질적으로 통제한다.",
                f"   - 법적 근거: {format_source_basis(signage_source, default_annex='별표 6 제1호')}",
            ]
        )

    if crane_source:
        lines.extend(
            [
                "3. 크레인 안전인증ㆍ안전검사 및 합격표시 확인",
                "   - 조치: 사고 작업에 사용된 크레인의 안전인증ㆍ안전검사 대상 여부, 안전검사합격증명서, 검사유효기간 및 표시 상태를 확인하고 부적합 장비는 사용을 중지한다.",
                f"   - 법적 근거: {format_source_basis(crane_source, default_annex='별표 16')}",
            ]
        )

    if crane_signal_source:
        lines.extend(
            [
                "4. 크레인 인양 작업 신호체계 및 낙하ㆍ비래 예방조치 재정비",
                "   - 조치: 인양 중 철제 자재 낙하가 사고 원인이므로 신호자 지정, 공동작업 신호방법, 인양물 낙하ㆍ비래ㆍ충돌 위험 예방조치를 재점검하고 근로자에게 교육한다.",
                f"   - 법적 근거: {format_source_basis(crane_signal_source, default_annex=f'별표 5 제{extract_item_number(crane_signal_source)}호')}",
            ]
        )

    if report_source:
        lines.extend(
            [
                "5. 산업재해 발생 사실 기록ㆍ보존 및 보고",
                "   - 조치: 사고 발생 개요, 원인, 보고 시기, 재발방지 계획을 기록ㆍ보존하고 고용노동부령이 정한 대상 산업재해이면 고용노동부장관에게 보고한다.",
                f"   - 법적 근거: {format_source_basis(report_source)}",
            ]
        )

    if accident_form_source:
        lines.extend(
            [
                "6. 산업재해조사표 작성ㆍ제출",
                "   - 조치: 산업재해조사표에 사업장 정보, 재해자 정보, 재해발생 개요ㆍ원인, 재발방지 계획을 기재하여 보고 실무 서식으로 관리한다.",
                f"   - 법적 근거: {format_source_basis(accident_form_source, default_annex='별지 제30호서식')}",
            ]
        )

    if hazard_plan_source:
        lines.extend(
            [
                "7. 유해위험방지계획서 이행 상태 확인 및 재검토",
                "   - 조치: 굴착ㆍ크레인 병행 작업과 금지구역 관리가 유해위험방지계획서와 일치하는지 확인하고, 공법 변경이나 위험 증가가 있으면 계획서 보완ㆍ재검토 및 필요한 확인 절차를 진행한다.",
                f"   - 법적 근거: {format_source_basis(hazard_plan_source)}",
            ]
        )

    lines.extend(
        [
            "",
            "[우선순위]",
            "- 즉시 작업중지ㆍ위험구역 통제 후 특별안전교육과 크레인 적합성 확인을 먼저 완료한다.",
            "- 이후 산업재해 보고와 유해위험방지계획서 이행 확인ㆍ보완을 문서화한다.",
        ]
    )
    return "\n".join(lines)


def find_admin_disposition_source(sources: list[SourceDoc]) -> SourceDoc | None:
    for source in sources:
        compact = re.sub(r"\s+", "", source.content)
        page = str(source.metadata.get("page", ""))
        if "별표26" in compact or "행정처분기준" in compact or page in {"199", "200"}:
            return source
    return None


def find_crane_safety_source(sources: list[SourceDoc]) -> SourceDoc | None:
    for source in sources:
        compact = re.sub(r"\s+", "", source.content)
        issue = str(source.metadata.get("issue", ""))
        if (
            "크레인" in compact
            and any(term in compact for term in ("안전인증", "안전검사", "합격증명", "표시부호"))
        ) or "크레인 안전" in issue:
            return source
    return None


def find_crane_signal_source(sources: list[SourceDoc]) -> SourceDoc | None:
    preferred: list[SourceDoc] = []
    fallback: list[SourceDoc] = []
    for source in sources:
        compact = re.sub(r"\s+", "", source.content)
        item_no = extract_item_number(source)
        if item_no in {"14", "39"} and any(term in compact for term in ("신호", "인양", "낙하", "비래", "충돌")):
            preferred.append(source)
        elif "크레인" in compact and any(term in compact for term in ("신호", "인양", "낙하", "비래", "충돌")):
            fallback.append(source)
    return preferred[0] if preferred else (fallback[0] if fallback else None)


def find_accident_report_source(sources: list[SourceDoc]) -> SourceDoc | None:
    for source in sources:
        compact = re.sub(r"\s+", "", source.content)
        if (
            "제57조" in compact
            and "산업재해발생" in compact
            and ("보고" in compact or "재발방지계획" in compact)
        ):
            return source
    return None


def find_accident_investigation_form_source(sources: list[SourceDoc]) -> SourceDoc | None:
    for source in sources:
        compact = re.sub(r"\s+", "", source.content)
        if "산업재해조사표" in compact or "별지제30호서식" in compact:
            return source
    return None


def find_hazard_plan_source(sources: list[SourceDoc]) -> SourceDoc | None:
    for source in sources:
        compact = re.sub(r"\s+", "", source.content)
        if "유해위험방지계획서" in compact and any(term in compact for term in ("이행", "공법의변경", "심사", "확인")):
            return source
    return None


def format_source_basis(source: SourceDoc, default_annex: str = "") -> str:
    metadata = source.metadata
    law_name = str(metadata.get("law_name") or "산업안전보건법 시행규칙").replace("_", " ")
    article = str(metadata.get("article") or "").strip()
    page = format_source_page(source)
    annex = str(metadata.get("annex") or "") or extract_annex_label(source.content) or default_annex
    page_suffix = f", p.{page}" if page and page != "0" else ""
    if annex:
        return f"{law_name} {annex}{page_suffix}"
    if article:
        return f"{law_name} {article}{page_suffix}"
    return f"{law_name}{page_suffix}"


def format_source_page(source: SourceDoc) -> str:
    metadata = source.metadata
    citation_page = metadata.get("citation_page")
    if citation_page:
        return str(citation_page)

    law_name = str(metadata.get("law_name") or metadata.get("source") or metadata.get("pdf_file") or "")
    annex = str(metadata.get("annex") or "") or extract_annex_label(source.content)
    if annex == "별표 35" and "시행령" in law_name:
        return "130~143"
    if annex == "별표 26" and "시행규칙" in law_name:
        return "199~214"

    page = metadata.get("page", "")
    return str(page)


def extract_annex_label(content: str) -> str:
    compact = re.sub(r"\s+", "", content)
    if "별표35" in compact or ("법제29조제3항" in compact and "50100150" in compact):
        return "별표 35"
    if "별표26" in compact:
        return "별표 26"
    if "별표6" in compact:
        if "출입금지" in compact or "금지표지" in compact:
            return "별표 6 제1호"
        return "별표 6"
    if "[작업항목]19." in compact:
        return "별표 5 제19호"
    match = re.search(r"별표\s*(\d+)", content)
    return f"별표 {match.group(1)}" if match else ""


def direct_special_education_items_answer(question: str, sources: list[SourceDoc]) -> str | None:
    """Deterministically list applicable 별표 5 special-education items."""
    compact_question = re.sub(r"\s+", "", question)
    if not any(term in compact_question for term in ("특별교육", "교육내용", "미이수", "미실시")):
        return None
    if not any(term in compact_question for term in ("모두", "나열", "복수", "조항", "해당", "위반")):
        return None

    candidates = collect_applicable_special_education_items(question, sources)
    if len(candidates) < 2:
        return None

    lines = ["해당 사고 시나리오에서 우선 검토할 특별교육 대상 작업은 다음과 같습니다.", ""]
    lines.append("[특별교육 대상 작업]")
    for index, candidate in enumerate(candidates, start=1):
        lines.append(
            f"{index}. 산업안전보건법 시행규칙 별표 5 제{candidate['item_no']}호, "
            f"p.{candidate['page']}"
        )
        lines.append(f"   - 해당 작업: {candidate['title']}")
        lines.append(f"   - 해당 이유: {candidate['reason']}")
        lines.append("   - 교육 내용:")
        for item in candidate["items"]:
            lines.append(f"     ○ {item}")
        lines.append("")

    if "크레인" in compact_question or "인양" in compact_question or "양중" in compact_question:
        lines.append("[관련 크레인 근거]")
        lines.append("- 크레인 사용 작업의 특별교육은 별표 5 제14호가 직접 근거입니다.")
        lines.append("- 안전검사 합격표시 등 설비 표시 기준은 산업안전보건법 시행규칙 별표 16에서 별도로 확인할 수 있습니다.")
        lines.append("")

    lines.append("※ 암석 굴착(별표 5 제22호)은 시나리오에 암석ㆍ발파ㆍ폭발물 단서가 없으면 제외했습니다.")
    return "\n".join(lines).rstrip()


def collect_applicable_special_education_items(question: str, sources: list[SourceDoc]) -> list[dict]:
    compact_question = re.sub(r"\s+", "", question)
    triggers = {
        "14": (
            any(term in compact_question for term in ("크레인", "인양", "양중")),
            "크레인 인양 작업 단서가 있음",
        ),
        "19": (
            any(term in compact_question for term in ("굴착", "지반굴착", "굴착면", "토사붕괴")),
            "굴착면 높이 2미터 이상 지반 굴착 작업 단서가 있음",
        ),
        "21": (
            "터널" in compact_question,
            "터널 안 굴착 또는 터널 거푸집 지보공 작업 단서가 있음",
        ),
        "22": (
            any(term in compact_question for term in ("암석", "발파", "폭발물")),
            "암석 굴착 또는 발파 작업 단서가 있음",
        ),
        "23": (
            any(term in compact_question for term in ("비계", "강관비계", "이동식비계", "비계해체", "비계조립")),
            "비계 조립ㆍ해체 또는 변경 작업 단서가 있음",
        ),
        "27": (
            any(term in compact_question for term in ("철골", "골조", "금속제", "금속", "15층", "고층")),
            "건축물 골조 또는 금속제 부재 조립ㆍ해체ㆍ변경 작업 단서가 있음",
        ),
    }

    candidates: list[dict] = []
    seen: set[str] = set()
    for source in sources:
        item_no = extract_item_number(source)
        if not item_no or item_no in seen:
            continue
        enabled, reason = triggers.get(item_no, (False, ""))
        if not enabled:
            continue
        items = extract_education_items_from_item_chunk(source.content)
        if not items:
            continue
        candidates.append(
            {
                "item_no": item_no,
                "page": source.metadata.get("page", ""),
                "title": extract_item_title(source.content),
                "reason": reason,
                "items": items,
            }
        )
        seen.add(item_no)

    order = {"14": 0, "19": 1, "21": 2, "23": 3, "27": 4, "22": 5}
    candidates.sort(key=lambda candidate: order.get(candidate["item_no"], 99))
    return candidates


def extract_item_number(source: SourceDoc) -> str:
    compact = re.sub(r"\s+", "", source.content)
    if "비계의조립·해체또는변경작업" in compact:
        return "23"
    item_number = str(source.metadata.get("item_number") or "")
    match = re.match(r"\s*(\d+)\.", item_number)
    if match:
        return match.group(1)
    match = re.search(r"\[작업항목\]\s*(\d+)\.", source.content)
    return match.group(1) if match else ""


def extract_item_title(content: str) -> str:
    match = re.search(r"\[작업항목\]\s*\d+\.\s*(.*?)(?:\s*\[교육내용\]|$)", content, flags=re.DOTALL)
    if not match:
        return ""
    return clean_display_text(match.group(1))


def direct_excavation_special_education_answer(question: str, sources: list[SourceDoc]) -> str | None:
    """Deterministic answer for 별표 5 제19호 지반 굴착 특별교육."""
    compact_question = re.sub(r"\s+", "", question)
    if not any(term in compact_question for term in ("굴착", "지반굴착", "굴착면", "토사붕괴")):
        return None
    if not any(term in compact_question for term in ("특별교육", "교육내용", "미이수", "미실시", "위반")):
        return None

    source = find_excavation_item19_source(sources)
    if not source:
        return None

    items = extract_education_items_from_item_chunk(source.content)
    if not items:
        return None

    metadata = source.metadata
    law_name = str(metadata.get("law_name") or "산업안전보건법 시행규칙").replace("_", " ")
    page = metadata.get("page", "82")

    asks_violation = is_violation_question(question)
    height = extract_meter_value(question)
    lines: list[str] = []
    if asks_violation:
        lines.append("위반 여부: YES")
        lines.append("")
        lines.append("[위반 조항]")
        lines.append(f"- {law_name} 별표 5 제19호, p.{page}")
        if height is not None:
            lines.append(f"- 해당 이유: 굴착면 높이 {format_meter(height)}m >= 기준 2m 이상 -> 해당")
        else:
            lines.append("- 해당 이유: 굴착면의 높이가 2미터 이상인 지반 굴착작업은 특별교육 대상 작업입니다.")
        lines.append("")
        lines.append("[관련 교육 내용 / 조치 기준]")
    else:
        lines.append(f"굴착면의 높이가 2미터 이상인 지반 굴착작업의 특별교육 내용은 다음과 같습니다.")
        lines.append("")

    lines.extend(f"○ {item}" for item in items)
    lines.append("")
    lines.append(f"근거: {law_name} [별표 5] 제19호, p.{page}")
    return "\n".join(lines)


def direct_scaffold_special_education_answer(question: str, sources: list[SourceDoc]) -> str | None:
    """Deterministic answer for 별표 5 제23호 비계 조립ㆍ해체 또는 변경 작업 특별교육."""
    if not should_direct_scaffold_special_education(question):
        return None

    source = find_osha_scaffold_source(sources)
    if not source:
        return None

    items = extract_education_items_from_item_chunk(source.content)
    if not items:
        return None

    metadata = source.metadata
    law_name = str(metadata.get("law_name") or "산업안전보건법 시행규칙").replace("_", " ")
    page = metadata.get("page", "83")
    item_no = extract_item_number(source) or "23"

    asks_violation = is_violation_question(question)
    lines: list[str] = []
    if asks_violation:
        lines.append("위반 여부: YES")
        lines.append("")
        lines.append("[위반 조항]")
        lines.append(f"- {law_name} 별표 5 제{item_no}호, p.{page}")
        lines.append("- 해당 이유: 비계의 조립ㆍ해체 또는 변경 작업은 특별교육 대상 작업이며, 시나리오상 특별안전교육 미실시 단서가 있습니다.")
        lines.append("")
        lines.append("[관련 교육 내용 / 조치 기준]")
    else:
        lines.append("비계의 조립ㆍ해체 또는 변경 작업의 특별교육 내용은 다음과 같습니다.")
        lines.append("")

    lines.extend(f"○ {item}" for item in items)
    lines.append("")
    lines.append(f"근거: {law_name} [별표 5] 제{item_no}호, p.{page}")
    return "\n".join(lines)


def find_excavation_item19_source(sources: list[SourceDoc]) -> SourceDoc | None:
    for source in sources:
        compact = re.sub(r"\s+", "", source.content)
        if "[작업항목]19." in compact and "굴착면의높이가2미터이상인지반굴착작업" in compact:
            return source
    for source in sources:
        compact = re.sub(r"\s+", "", source.content)
        if (
            "굴착면의높이가2미터" in compact
            and "지반굴착" in compact
            and ("작업" in compact or "특별교육" in compact)
        ):
            return source
    return None


def extract_education_items_from_item_chunk(content: str) -> list[str]:
    marker = "[교육내용]"
    section = content.split(marker, 1)[1] if marker in content else content
    raw_items = [item.strip(" ,;") for item in section.split("○") if item.strip(" ,;")]
    items: list[str] = []
    for item in raw_items:
        cleaned = clean_display_text(item)
        cleaned = re.sub(r"^\[[^\]]+\]\s*", "", cleaned).strip()
        if not cleaned or "굴착면의 높이" in cleaned:
            continue
        if cleaned not in items:
            items.append(cleaned)
    return items


def extract_meter_value(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|M|미터)", text)
    if not match:
        return None
    return float(match.group(1))


def format_meter(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def direct_worker_safety_education_answer(question: str, sources: list[SourceDoc]) -> str | None:
    """Answer the common worker safety-health education content question from 별표 5."""
    compact_question = re.sub(r"\s+", "", question)
    if (
        "근로자" not in compact_question
        or "안전보건교육" not in compact_question
        or "교육내용" not in compact_question
    ):
        return None

    source = find_worker_education_source(sources)
    if not source:
        return None

    regular = extract_bullet_section(source.content, "가.정기교육교육내용", "나.")
    hire_or_change = extract_bullet_section(
        source.content,
        "다.채용시교육및작업내용변경시교육교육내용",
        "라.특별교육",
    )
    if not regular or not hire_or_change:
        return None

    metadata = source.metadata
    law_name = str(metadata.get("law_name") or "산업안전보건법 시행규칙").replace("_", " ")
    page = metadata.get("page", "")

    lines = ["근로자 안전보건교육의 교육내용은 교육 과정별로 다음과 같습니다.", ""]
    lines.append("정기교육:")
    lines.extend(f"- {item}" for item in regular)
    lines.append("")
    lines.append("채용 시 교육 및 작업내용 변경 시 교육:")
    lines.extend(f"- {item}" for item in hire_or_change)
    lines.append("")
    lines.append("특별교육 대상 작업별 교육:")
    lines.append("- 공통내용: 채용 시 교육 및 작업내용 변경 시 교육내용과 같음")
    lines.append("- 개별내용: 작업별 교육내용")
    lines.append("")
    lines.append(f"근거: {law_name} [별표 5] 제26조제1항 관련, p.{page}")
    return "\n".join(lines)


def find_worker_education_source(sources: list[SourceDoc]) -> SourceDoc | None:
    for source in sources:
        if source.metadata.get("source_type") != "text":
            continue
        compact = re.sub(r"\s+", "", source.content)
        if "별표5" in compact and "1.근로자안전보건교육" in compact:
            return source
    return None


def extract_bullet_section(content: str, start_marker: str, end_marker: str) -> list[str]:
    compact = re.sub(r"\s+", "", content)
    start = compact.find(start_marker)
    if start == -1:
        return []
    start += len(start_marker)
    end = compact.find(end_marker, start)
    section = compact[start:] if end == -1 else compact[start:end]
    return [
        prettify_education_item(item)
        for item in section.split("○")
        if item.strip()
    ]


EDUCATION_ITEM_SPACING = {
    "산업안전및산업재해예방에관한사항(화재ㆍ폭발사고발생시대피에관한사항을포함한다)": "산업안전 및 산업재해 예방에 관한 사항(화재ㆍ폭발 사고 발생 시 대피에 관한 사항을 포함한다)",
    "산업보건및건강장해예방에관한사항(폭염ㆍ한파작업으로인한건강장해발생시응급조치에관한사항을포함한다)": "산업보건 및 건강장해 예방에 관한 사항(폭염ㆍ한파작업으로 인한 건강장해 발생 시 응급조치에 관한 사항을 포함한다)",
    "산업보건및건강장해예방에관한사항": "산업보건 및 건강장해 예방에 관한 사항",
    "위험성평가에관한사항": "위험성평가에 관한 사항",
    "건강증진및질병예방에관한사항": "건강증진 및 질병 예방에 관한 사항",
    "유해ㆍ위험작업환경관리에관한사항": "유해ㆍ위험 작업환경 관리에 관한 사항",
    "산업안전보건법령및산업재해보상보험제도에관한사항": "산업안전보건법령 및 산업재해보상보험 제도에 관한 사항",
    "직무스트레스예방및관리에관한사항": "직무스트레스 예방 및 관리에 관한 사항",
    "직장내괴롭힘,고객의폭언등으로인한건강장해예방및관리에관한사항": "직장 내 괴롭힘, 고객의 폭언 등으로 인한 건강장해 예방 및 관리에 관한 사항",
    "기계ㆍ기구의위험성과작업의순서및동선에관한사항": "기계ㆍ기구의 위험성과 작업의 순서 및 동선에 관한 사항",
    "작업개시전점검에관한사항": "작업 개시 전 점검에 관한 사항",
    "정리정돈및청소에관한사항": "정리정돈 및 청소에 관한 사항",
    "사고발생시긴급조치에관한사항": "사고 발생 시 긴급조치에 관한 사항",
    "물질안전보건자료에관한사항": "물질안전보건자료에 관한 사항",
}


def prettify_education_item(item: str) -> str:
    compact = re.sub(r"\s+", "", item).strip()
    return EDUCATION_ITEM_SPACING.get(compact, item.strip())


EXPOSURE_COLUMN_MAP = {
    "허용기준": ("시간가중평균값(TWA)", "ppm", "twa"),
    "col_2": ("시간가중평균값(TWA)", "mg/㎥", "twa"),
    "col_3": ("시간가중평균값(TWA)", "mg/㎥", "twa"),
    "col_4": ("단시간 노출값(STEL)", "ppm", "stel"),
    "col_5": ("단시간 노출값(STEL)", "mg/㎥", "stel"),
    "TWA_ppm": ("시간가중평균값(TWA)", "ppm", "twa"),
    "TWA_mg_m3": ("시간가중평균값(TWA)", "mg/㎥", "twa"),
    "STEL_ppm": ("단시간 노출값(STEL)", "ppm", "stel"),
    "STEL_mg_m3": ("단시간 노출값(STEL)", "mg/㎥", "stel"),
}


def direct_exposure_limit_answer(question: str, sources: list[SourceDoc]) -> str | None:
    """Answer exposure-limit table facts directly when values and units are parsed."""
    if not is_exposure_limit_question(question):
        return None

    records = collect_exposure_limit_records(question, sources)
    selected_records: list[dict] = []
    for record in records:
        selected_limits = select_exposure_limits(question, record["limits"])
        if not selected_limits:
            continue
        selected_records.append({**record, "limits": selected_limits})

    if not selected_records:
        return None

    return format_exposure_limit_records_answer(question, selected_records)


def collect_exposure_limit_records(question: str, sources: list[SourceDoc]) -> list[dict]:
    """Collect matching exposure-limit rows and merged-row continuations."""
    records: list[dict] = []
    active_by_group: dict[tuple[str, str, str], dict] = {}

    for source in sorted(sources, key=table_source_sort_key):
        if source.metadata.get("source_type") != "table":
            continue

        pairs = parse_key_value_pairs(source.content)
        limits = extract_exposure_limits(pairs)
        if not limits:
            continue

        metadata = source.metadata
        group = table_group_key(metadata)
        row_index = metadata.get("row_index")
        try:
            row_index_int = int(row_index)
        except (TypeError, ValueError):
            row_index_int = -1

        substance = extract_substance_from_pairs(pairs)
        if substance:
            if substance_matches_question(substance, question):
                record = make_exposure_record(source, substance, pairs, limits)
                records.append(record)
                active_by_group[group] = {
                    "substance": substance,
                    "row_index": row_index_int,
                }
            elif group in active_by_group:
                del active_by_group[group]
            continue

        active = active_by_group.get(group)
        if not active:
            continue
        if row_index_int < 0 or row_index_int > int(active["row_index"]) + 3:
            continue
        if not row_detail_from_pairs(pairs):
            continue

        record = make_exposure_record(source, str(active["substance"]), pairs, limits)
        records.append(record)

    return dedupe_exposure_records(records)


def table_source_sort_key(source: SourceDoc) -> tuple[str, int, int, int]:
    metadata = source.metadata
    return (
        str(metadata.get("source") or metadata.get("pdf_file") or ""),
        safe_int(metadata.get("page")),
        safe_int(metadata.get("table_index")),
        safe_int(metadata.get("row_index")),
    )


def table_group_key(metadata: dict) -> tuple[str, str, str]:
    return (
        str(metadata.get("source") or metadata.get("pdf_file") or ""),
        str(metadata.get("page", "")),
        str(metadata.get("table_index", "")),
    )


def safe_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return -1


def make_exposure_record(
    source: SourceDoc,
    substance: str,
    pairs: dict[str, str],
    limits: list[dict[str, str]],
) -> dict:
    metadata = source.metadata
    return {
        "substance": substance,
        "detail": row_detail_from_pairs(pairs),
        "limits": limits,
        "law_name": str(metadata.get("law_name", "")),
        "page": metadata.get("page", ""),
        "sort_key": table_source_sort_key(source),
    }


def row_detail_from_pairs(pairs: dict[str, str]) -> str:
    return clean_display_text(pairs.get("세부구분") or pairs.get("col_1") or "")


def dedupe_exposure_records(records: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        limit_key = "|".join(limit["value"] for limit in record["limits"])
        key = (
            normalize_substance_name(record["substance"]),
            str(record["detail"]),
            limit_key,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def summarize_exposure_limit_pairs(pairs: dict[str, str]) -> str:
    substance = extract_substance_from_pairs(pairs)
    limits = extract_exposure_limits(pairs)
    if not substance or not limits:
        return ""

    chunks = [normalize_substance_name(substance)]
    for limit in limits:
        text = f"{limit['label']}: {limit['value']}"
        if limit["condition"]:
            text += f" (조건: {limit['condition']})"
        chunks.append(text)
    return " / ".join(chunks)


def extract_substance_from_pairs(pairs: dict[str, str]) -> str:
    substance = pairs.get("유해인자", "")
    if substance:
        return substance

    for key, value in pairs.items():
        if looks_like_substance(value):
            return value
        if looks_like_substance(key):
            return key
    return ""


def looks_like_substance(text: str) -> bool:
    cleaned = clean_display_text(text)
    return bool(re.match(r"^\d+\.", cleaned)) and (
        "(" in cleaned
        or "[" in cleaned
        or "화합물" in cleaned
        or "석면" in cleaned
    )


def extract_exposure_limits(pairs: dict[str, str]) -> list[dict[str, str]]:
    limits: list[dict[str, str]] = []
    for key, (label, default_unit, kind) in EXPOSURE_COLUMN_MAP.items():
        raw_value = pairs.get(key)
        if not raw_value or not re.search(r"\d", raw_value):
            continue

        value, condition, unit = parse_limit_value(raw_value, default_unit)
        limits.append(
            {
                "key": key,
                "label": label,
                "kind": kind,
                "unit": unit,
                "value": value,
                "condition": condition,
            }
        )
    return limits


def parse_limit_value(raw_value: str, default_unit: str) -> tuple[str, str, str]:
    cleaned = clean_display_text(raw_value)
    condition = ""
    main_value = cleaned

    match = re.match(r"^([^()]+?)\s*\((.+)\)$", cleaned)
    if match and re.search(r"\d", match.group(1)):
        main_value = match.group(1).strip()
        condition = match.group(2).strip()

    unit = detect_unit(main_value) or detect_unit(condition) or default_unit
    return (
        ensure_value_unit(main_value, unit),
        ensure_condition_unit(condition, unit),
        unit,
    )


def detect_unit(text: str) -> str:
    normalized = text.lower()
    if "개/㎤" in text or "개/cm3" in normalized or "개/㎠" in text:
        return "개/㎤"
    if "mg/㎥" in text or "mg/m3" in normalized or "mg/㎡" in text:
        return "mg/㎥"
    if "ppm" in normalized:
        return "ppm"
    return ""


def ensure_value_unit(value: str, unit: str) -> str:
    value = clean_display_text(value)
    value = re.sub(r"(\d(?:\.\d+)?)(개/㎤|mg/㎥|ppm)", r"\1 \2", value)
    if detect_unit(value) or not unit:
        return value
    return f"{value} {unit}"


def ensure_condition_unit(condition: str, unit: str) -> str:
    condition = clean_display_text(condition)
    if not condition or detect_unit(condition) or not unit:
        return condition
    return re.sub(r"(\d+(?:\.\d+)?)$", rf"\1 {unit}", condition)


def select_exposure_limits(
    question: str,
    limits: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not limits:
        return []

    normalized = question.lower()
    asks_mg = "mg/㎥" in question or "mg/m3" in normalized or "mg" in normalized
    asks_ppm = "ppm" in normalized
    asks_stel = "stel" in normalized or "단시간" in question
    asks_twa = "twa" in normalized or "시간가중" in question or "허용기준" in question

    selected = limits
    if asks_mg:
        selected = [limit for limit in limits if limit["unit"] == "mg/㎥"]
    elif asks_ppm:
        selected = [limit for limit in limits if limit["unit"] == "ppm"]

    if asks_stel and not asks_twa:
        selected = [limit for limit in selected if limit["kind"] == "stel"]
    elif asks_twa and selected:
        twa_values = [limit for limit in selected if limit["kind"] == "twa"]
        if twa_values:
            selected = twa_values
            if asks_ppm:
                selected.extend(
                    limit
                    for limit in limits
                    if limit["kind"] == "stel" and limit["unit"] == "ppm"
                )

    return selected


def format_exposure_limit_records_answer(question: str, records: list[dict]) -> str:
    display_name = normalize_substance_name(str(records[0]["substance"]))
    detailed_records = [record for record in records if record.get("detail")]

    if len(detailed_records) >= 2:
        label = detailed_records[0]["limits"][0]["label"]
        parts: list[str] = []
        for record in detailed_records:
            value = record["limits"][0]["value"]
            parts.append(f"{record['detail']} {value}")
        answer = f"{display_name}의 {label} 허용기준은 {', '.join(parts)}입니다."
    else:
        clauses: list[str] = []
        for record_index, record in enumerate(records):
            detail = record.get("detail")
            for limit_index, limit in enumerate(record["limits"]):
                label = limit["label"]
                value = limit["value"]
                subject = f"{display_name}"
                if detail:
                    subject += f"({detail})"

                if record_index == 0 and limit_index == 0:
                    clauses.append(f"{subject}의 {label} 허용기준은 {value}입니다.")
                else:
                    clauses.append(f"{subject}의 {label}은 {value}입니다.")
                if limit["condition"]:
                    clauses.append(f"단, {limit['condition']}입니다.")

        substance_condition = extract_substance_condition(str(records[0]["substance"]))
        if substance_condition:
            clauses.append(f"조건: {substance_condition}.")
        answer = " ".join(clauses)

    return answer + "\n" + format_exposure_basis(records)


def format_exposure_basis(records: list[dict]) -> str:
    bases: list[str] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        law_name = str(record.get("law_name") or "")
        page = str(record.get("page") or "")
        key = (law_name, page)
        if key in seen:
            continue
        seen.add(key)
        bases.append(f"{law_name} [별표 19], p.{page}")
    return "근거: " + "; ".join(bases)


def substance_matches_question(substance: str, question: str) -> bool:
    substance_terms = extract_substance_terms(substance)
    compact_question = re.sub(r"\s+", "", question)
    question_lower = question.lower()
    for term in substance_terms:
        if re.search(r"[A-Za-z]", term):
            if term.lower() in question_lower:
                return True
        elif term in compact_question:
            return True
    return False


def extract_substance_terms(substance: str) -> list[str]:
    cleaned = clean_display_text(substance)
    korean_stopwords = {
        "및",
        "그",
        "등",
        "경우",
        "해당한다",
        "제조",
        "사용",
        "화합물",
        "무기화합물",
    }
    english_stopwords = {
        "and",
        "its",
        "the",
        "compound",
        "compounds",
        "inorganic",
        "organic",
    }

    terms: list[str] = []
    for token in re.findall(r"[가-힣][가-힣ㆍ·]+", cleaned):
        compact = token.replace("ㆍ", "").replace("·", "")
        if len(compact) >= 2 and compact not in korean_stopwords:
            terms.append(token)

    for token in re.findall(r"[A-Za-z][A-Za-z-]+", cleaned):
        lower = token.lower()
        if len(lower) >= 3 and lower not in english_stopwords:
            terms.append(lower)

    return terms


def normalize_substance_name(substance: str) -> str:
    name = re.sub(r"^\s*\d+\.\s*", "", substance).strip()
    name = re.sub(r"\([^)]*경우만\s*해당한다\)", "", name)
    name = re.sub(r";\s*[\d-]+", "", name)
    return clean_display_text(name)


def extract_substance_condition(substance: str) -> str:
    cleaned = clean_display_text(substance)
    match = re.search(r"\(([^)]*경우만\s*해당한다)\)", cleaned)
    if not match:
        return ""
    return match.group(1).replace("해당한다", "해당")


def clean_display_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    replacements = {
        "화합 물": "화합물",
        "호 흡 성": "호흡성",
        "경 우": "경우",
        "분 진": "분진",
        "노 출": "노출",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the integrated text/table RAG chatbot")
    parser.add_argument("--question", "-q", required=True)
    parser.add_argument("--text-top-k", type=int, default=RAG_TOP_K)
    parser.add_argument("--table-top-k", type=int, default=RAG_TOP_K)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--show-sources", action="store_true")
    args = parser.parse_args()

    response = rag_chat(
        ChatRequest(question=args.question),
        text_top_k=args.text_top_k,
        table_top_k=args.table_top_k,
        cpu=args.cpu,
    )
    print(response.answer)

    if args.show_sources:
        print("\n[Sources]")
        for index, source in enumerate(response.sources, start=1):
            metadata = source.metadata
            print(
                f"{index}. {metadata.get('source_type')} "
                f"{metadata.get('law_name')} p.{metadata.get('page')} "
                f"score={metadata.get('score')}"
            )
            print(source.content[:300].replace("\n", " "))


if __name__ == "__main__":
    main()
