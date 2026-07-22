from rag.chatbot import build_retrieval_query, direct_answer_from_sources, direct_answer_sources
from rag.citation_validator import validate_answer_citations
from rag.industrial_fire_routing import classify_lithium_question, is_lithium_battery_fire_scenario
from rag.question_graph import run_question_graph
from rag.scenario_analysis import validate_scenario_profile
from rag.schemas import AccidentScenario


SCENARIO = AccidentScenario(
    overview="(주)한빛에너지셀 리튬전지 공장에서 전지 폭발과 연쇄 화재로 근로자 23명이 사망하고 8명이 부상했다.",
    details=(
        "전지 3만5천개가 적재되어 있었고 유사 발열ㆍ화재 전조가 5차례 있었지만 작업을 계속했다. "
        "발열검사와 분리보관 절차가 없었고 위험성평가는 형식적으로 작성되었다. "
        "비상구와 대피통로가 미흡했고 안전보건교육과 소방훈련도 실시하지 않았다."
    ),
    workers="이주노동자는 인력공급업체를 통해 투입되어 도급ㆍ파견이 혼재했다. 사망자: 23명, 부상자: 8명",
)


QUESTIONS = {
    "lithium_hazard_training": "리튬전지 위험물질 취급과 안전보건교육 미실시의 위반 여부를 판단하라.",
    "lithium_mass_fatality_sentencing": "23명 다수사망이 중대재해처벌법상 처벌 수위와 양형에 미치는 영향을 판단하라.",
    "lithium_contract_dispatch": "이주노동자가 파견ㆍ도급 형태로 투입된 경우 한빛에너지셀의 책임과 각 법령의 근거를 제시하라.",
    "lithium_risk_warning": "형식적 위험성평가와 5차례 화재 전조를 무시한 것이 경영책임자 의무 위반인지 판단하라.",
    "lithium_comprehensive": "사고 원인, 법령 위반, 책임 주체, 다수사망 양형, 재발방지 조치를 포함한 최종 보고서 형식으로 종합 평가하라.",
}


def _context(question: str) -> tuple[str, dict]:
    profile = validate_scenario_profile({}, SCENARIO)
    return build_retrieval_query(question, SCENARIO, profile), profile


def test_lithium_scenario_and_all_question_types_are_separated() -> None:
    assert is_lithium_battery_fire_scenario("리튬전지 발열 후 폭발 화재")
    for expected, question in QUESTIONS.items():
        context, _ = _context(question)
        assert classify_lithium_question(question, context) == expected


def test_question_graph_uses_profile_and_exposes_lithium_intent() -> None:
    question = QUESTIONS["lithium_contract_dispatch"]
    context, profile = _context(question)
    state = run_question_graph(question, mode="scenario", cache_context=context, scenario_profile=profile)

    assert state["scope"] == "scenario_judgment"
    assert state["intent"] == "lithium_contract_dispatch"
    assert state["route"] == "direct_candidate"
    assert state["scenario_profile_used"] is True
    assert "active_scenario_profile" in state["scope_signals"]


def test_lithium_q1_to_q5_never_reuse_legacy_scenario_facts() -> None:
    forbidden = ("스파일럿건설", "A건설", "외벽 보수", "비계 작업", "굴착구역", "크레인")
    for kind, question in QUESTIONS.items():
        context, _ = _context(question)
        answer = direct_answer_from_sources(question, [], context, mode="scenario")
        sources = direct_answer_sources(question, [], context)

        assert answer
        assert all(term not in answer for term in forbidden)
        assert validate_answer_citations(answer, sources)["status"] == "pass"
        if kind == "lithium_hazard_training":
            assert "제225조" in answer and "제232조" in answer and "제29조" in answer
        elif kind == "lithium_mass_fatality_sentencing":
            assert "법정형 상한이 자동 상향" in answer
            assert "5년 이내 재범" in answer
        elif kind == "lithium_contract_dispatch":
            assert "(주)한빛에너지셀" in answer
            assert "한빛에너지셀의의" not in answer
            assert "제63조" in answer and "제64조" in answer and "제5조" in answer
        elif kind == "lithium_risk_warning":
            assert "5차례" in answer and "제36조" in answer and "제51조" in answer
        elif kind == "lithium_comprehensive":
            assert "사망자 23명" in answer
            assert "현재 자료에서 구체적 감경 사실은 확인되지 않습니다" in answer
