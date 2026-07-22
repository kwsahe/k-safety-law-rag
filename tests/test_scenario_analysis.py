from rag.chatbot import build_retrieval_query
from rag.question_graph import run_question_graph
from rag.scenario_analysis import scenario_hash, validate_scenario_profile
from rag.schemas import AccidentScenario


SCENARIO = AccidentScenario(
    overview="(주)신성종합건설 현장에서 조적 작업 중 벽돌이 떨어져 근로자 1명이 사망했다.",
    details="조적 작업은 전문업체 E사에 도급했고 발끝막이판과 낙하물 방지망이 없었다.",
    workers="사망자 1명",
)


def test_scenario_hash_changes_with_saved_content() -> None:
    changed = AccidentScenario(**{**SCENARIO.model_dump(), "workers": "부상자 1명"})
    assert scenario_hash(SCENARIO) != scenario_hash(changed)


def test_validation_corrects_routing_facts_from_scenario_text() -> None:
    profile = validate_scenario_profile(
        {
            "accident_type": "굴착",
            "company": "잘못된 회사",
            "contractor": "B사",
            "death_count": 0,
            "hazards": ["벽돌 낙하"],
            "missing_controls": ["낙하물 방지망"],
        },
        SCENARIO,
    )
    assert profile["accident_type"] == "물체맞음"
    assert profile["company"] == "(주)신성종합건설"
    assert profile["contractor"] == "E사"
    assert profile["death_count"] == 1
    assert profile["routing_hints"] == ["masonry_falling"]
    assert profile["validation"]["legal_articles_decided_by_llm"] is False


def test_validation_rejects_company_names_missing_from_source() -> None:
    scenario = AccidentScenario(overview="작업장에서 자재가 떨어졌다.")
    profile = validate_scenario_profile({"company": "(주)환각건설", "contractor": "Z사"}, scenario)
    assert profile["company"] == "확인 필요"
    assert profile["contractor"] == "확인 필요"


def test_fire_scenario_counts_and_contract_are_read_from_exact_phrases() -> None:
    scenario = AccidentScenario(
        overview="(주)한빛에너지셀에서 리튬전지 화재ㆍ폭발로 근로자 23명 사망, 8명 부상",
        details="유사한 전지 발열ㆍ화재가 5차례 있었으나 작업을 계속 진행하였다.",
        workers="사망자: 23명\n부상자: 8명\n직영 + 인력공급업체 도급/파견 혼재",
    )
    profile = validate_scenario_profile(
        {
            "injury_count": 23,
            "contract_structure": "확인 필요",
            "missing_controls": ["소방훈련 미실시"],
        },
        scenario,
    )
    assert profile["accident_type"] == "화재ㆍ폭발"
    assert profile["death_count"] == 23
    assert profile["injury_count"] == 8
    assert profile["long_term_injury_count"] is None
    assert profile["contract_structure"] == "도급/파견 혼재"
    assert profile["routing_hints"] == ["industrial_fire_explosion"]
    assert "반복 사고 전조 확인 후 작업중지 미실시" in profile["missing_controls"]


def test_profile_is_injected_before_question_scope() -> None:
    profile = validate_scenario_profile({}, SCENARIO)
    retrieval_query = build_retrieval_query("책임을 판단해줘.", SCENARIO, profile)
    state = run_question_graph(
        "책임을 판단해줘.",
        mode="scenario",
        cache_context=retrieval_query,
        scenario_profile=profile,
    )
    assert "[구조화 시나리오 분석]" in retrieval_query
    assert state["graph_nodes"][0] == "ScenarioProfileNode"
    assert state["cache_key"]
