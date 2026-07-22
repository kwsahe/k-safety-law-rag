from rag.chatbot import extract_accident_facts
from rag.report_payload import build_report_pages
from rag.schemas import SourceDoc
from rag.falling_object_routing import (
    direct_falling_controls_answer,
    direct_masonry_special_education_answer,
    is_masonry_falling_controls_question,
    is_masonry_falling_scenario,
    is_masonry_special_education_question,
)
from rag.v1_incident_routing import (
    _accident_outcome,
    _has_contract_relationship,
    direct_machine_controls_inspection_answer,
    direct_machine_inspection_answer,
    extract_company_name,
    extract_contractor_name,
    is_machine_entanglement_scenario,
    is_machine_controls_inspection_question,
    is_machine_inspection_question,
    is_struck_by_scenario,
)


def test_machine_and_struck_scenarios_are_separated() -> None:
    machine = "프레스 청소 중 방호장치를 해제해 근로자가 협착되었다."
    struck = "이동식 크레인 인양물이 낙하해 근로자가 맞았다."

    assert is_machine_entanglement_scenario(machine)
    assert not is_struck_by_scenario(machine)
    assert is_struck_by_scenario(struck)
    assert not is_machine_entanglement_scenario(struck)


def test_scenario_entity_names_are_dynamic() -> None:
    text = "원청 (주)동해플랜트가 수급업체 F사에 인양 작업을 도급했다."

    assert extract_company_name(text) == "(주)동해플랜트"
    assert extract_contractor_name(text) == "F사"


def test_report_payload_has_stable_page_contract() -> None:
    sources = [
        SourceDoc(
            content="산업안전보건법 제64조",
            metadata={"law_name": "산업안전보건법", "article": "제64조", "score": 0.98},
        )
    ]
    pages = build_report_pages("결론: 책임 성립\n- 위반 조항: 산업안전보건법 제64조", sources)

    assert [page["page"] for page in pages] == [7, 12]
    assert pages[0]["data"]["citations"]
    assert pages[1]["data"]["summary"] == "결론: 책임 성립"


def test_single_long_term_injury_is_not_a_serious_industrial_accident() -> None:
    scenario = "사망자: 없음 / 부상자: 1명 / T씨는 손가락 3개 절단으로 6개월 이상 치료가 필요하다."

    death, injury_count, treatment_months, applies = _accident_outcome(scenario)

    assert death is False
    assert injury_count == 1
    assert treatment_months == 6
    assert applies is False

    facts = extract_accident_facts(scenario)
    assert facts["death_count"] == 0
    assert facts["injury_count"] == 1
    assert facts["treatment_months"] == 6


def test_employment_months_are_not_used_as_treatment_months() -> None:
    scenario = "재해자 T씨 / 고용기간 4개월 / 손가락 3개 절단 / 6개월 이상 치료 필요 / 사망자: 없음 / 부상자: 1명"

    death, injury_count, treatment_months, applies = _accident_outcome(scenario)

    assert death is False
    assert injury_count == 1
    assert treatment_months == 6
    assert applies is False


def test_two_named_long_term_injuries_still_apply() -> None:
    scenario = "사망자는 없고 R씨와 S씨는 각각 9개월과 8개월 치료가 필요한 중상을 입었다."

    death, injury_count, treatment_months, applies = _accident_outcome(scenario)

    assert death is False
    assert injury_count == 2
    assert treatment_months == 8
    assert applies is True


def test_machine_inspection_question_is_separate_from_special_education() -> None:
    question = "프레스 안전검사와 자율안전확인 및 방호장치 관리 의무를 판단하라."

    assert is_machine_inspection_question(question)
    answer = direct_machine_inspection_answer([])
    assert "산업안전보건법 제93조" in answer
    assert "산업안전보건법 제89조" in answer
    assert "산업안전보건법 제80조제3항" in answer
    assert "특별교육" not in answer


def test_machine_controls_and_inspection_are_separate_from_self_confirmation() -> None:
    q2 = "프레스 방호장치 미작동, 안전검사 미실시, 정비 시 조치 위반을 각 항목별로 판단하라."
    q4 = "프레스 안전검사와 자율안전확인 및 방호장치 관리 의무를 판단하라."

    assert is_machine_controls_inspection_question(q2)
    assert not is_machine_inspection_question(q2)
    assert is_machine_inspection_question(q4)

    q2_answer = direct_machine_controls_inspection_answer([])
    q4_answer = direct_machine_inspection_answer([])
    assert q2_answer != q4_answer
    assert "제88조ㆍ제92조ㆍ제104조" in q2_answer
    assert "제조ㆍ수입자" not in q2_answer
    assert "제조ㆍ수입자" in q4_answer


def test_direct_work_is_not_classified_as_contracting() -> None:
    scenario = "도급 관계: 없음 (직영 작업)"

    assert _has_contract_relationship(scenario) is False


def test_masonry_falling_is_separate_from_crane_struck_by() -> None:
    scenario = "조적공이 비계 작업발판에 쌓아둔 벽돌이 낙하해 하부 근로자가 물체에 맞았다."

    assert is_masonry_falling_scenario(scenario)
    assert not is_struck_by_scenario(scenario)


def test_masonry_special_education_does_not_force_an_annex_item() -> None:
    scenario = "조적공의 벽돌 운반 중 자재가 낙하했다."
    question = "작업 관련 특별안전교육 미실시가 위반인가?"

    assert is_masonry_special_education_question(question, scenario)
    answer = direct_masonry_special_education_answer()
    assert "특별안전교육 미실시만으로" in answer
    assert "판단할 수 없습니다" in answer
    assert "별표 5 제19호" not in answer
    assert "별표 5 제23호" not in answer


def test_masonry_falling_controls_use_exact_provisions() -> None:
    question = "낙하물 방지망 미설치, 발끝막이판 미설치, 하부 출입통제 미실시를 판단하라."
    scenario = "조적 벽돌이 작업발판에서 낙하했다."

    assert is_masonry_falling_controls_question(question, scenario)
    answer = direct_falling_controls_answer()
    assert "제13조" in answer
    assert "제14조" in answer
    assert "제20조" in answer
    assert "안전대" not in answer
    assert "제56조" not in answer
