from rag.chatbot import (
    direct_answer_from_sources,
    make_osha_reference_source,
    verified_direct_response,
)
from rag.citation_validator import validate_answer_citations


LEGACY_TERMS = (
    "A건설",
    "B사",
    "C사",
    "D사",
    "(주)한국건설토건",
    "(주)한국제조기술",
    "(주)스파일럿건설",
    "지하 2층 굴착구역",
)


def test_legacy_serious_accident_templates_do_not_invent_scenario_facts() -> None:
    questions = (
        "중대재해처벌법상 경영책임자가 위반한 의무를 구체적으로 나열하라.",
        "중대재해처벌법상 도급 관계에서 원청 경영책임자의 책임을 판단하라.",
        "중대재해처벌법상 사고 후 대표이사 안전보건교육 과태료를 알려줘.",
    )
    for question in questions:
        answer = direct_answer_from_sources(question, [], mode="scenario")
        assert answer
        assert all(term not in answer for term in LEGACY_TERMS)


def test_synthetic_fallback_source_cannot_pass_citation_validation() -> None:
    source = make_osha_reference_source(
        "산업안전보건법",
        article="제64조",
        content="산업안전보건법 제64조 도급인의 산업재해 예방조치",
    )
    result = validate_answer_citations("근거: 산업안전보건법 제64조", [source])
    assert result["status"] == "warn"
    assert result["unsupported"]


def test_direct_answer_is_rejected_without_retrieved_db_evidence() -> None:
    question = (
        "도급 관계에서 원청의 책임이 성립하는지 산업안전보건법과 "
        "중대재해처벌법 각각의 근거를 제시하라."
    )
    response = verified_direct_response(
        question,
        [],
        question,
        mode="scenario",
    )
    assert response is None
