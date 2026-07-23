from rag.citation_validator import validate_answer_citations
from rag.general_law_routing import (
    classify_general_law_question,
    direct_general_law_answer,
    general_law_sources,
)


CASES = [
    ("중대재해처벌법과 산업안전보건법의 가장 큰 차이점은 무엇인가?", ("제38조", "제4조")),
    ("5인 이상 50인 미만 소규모 사업장도 중대재해처벌법이 전면 적용되는가?", ("5명 미만", "2024년 1월 27일")),
    ("중대재해처벌법상 '경영책임자등'의 범위는 어디까지인가?", ("제2조제9호", "실질적인 권한")),
    ("대표이사가 안전보건최고책임자(CSO)를 선임하면 형사처벌을 완전히 피할 수 있는가?", ("자동 면제", "제4조")),
    ("중대재해처벌법상 '중대산업재해'의 법적 정의 및 기준은 무엇인가?", ("사망자가 1명", "부상자가 2명", "질병자가 1년 이내 3명")),
    ("중대재해 발생 시 경영책임자에게 부과되는 처벌 수준은 어느 정도인가?", ("1년 이상", "10억원", "50억원", "7년 이하", "10억원 이하 벌금")),
    ("하도급(외주/협력업체) 근로자가 사고를 당했을 때도 원청 대표이사가 처벌받는가?", ("자동으로 처벌", "제5조", "제4조제9호")),
    ("중대재해처벌법 시행령상 '안전보건관리체계 구축'의 핵심 의무 항목은 무엇인가?", ("1.", "9.", "제4조")),
    ("위험성평가를 실시하지 않거나 형식적으로 작성하면 중대재해처벌법 위반인가?", ("제36조", "제4조제3호", "자동 확정")),
    ("산업안전보건법상 '안전보건관리책임자'와 중대재해처벌법상 '경영책임자'는 어떻게 다른가?", ("제15조", "제2조제9호", "사업장 단위")),
    ("중대재해 발생 시 기업이 부담해야 하는 '징벌적 손해배상' 규모는?", ("5배", "자동 부과")),
    ("안전보건 관계 법령에 따른 의무 이행 점검은 얼마나 자주 해야 하는가?", ("반기 1회", "제5조제2항제1호")),
    ("근로자가 위험을 느끼고 작업중지권을 행사할 때 사업주가 유의할 점은?", ("제52조제4항", "불리한 처우")),
    ("중대재해 발생 시 지체 없이 취해야 할 보고 및 대응 절차는?", ("제54조", "시행규칙 제67조", "지체 없이")),
    ("산안법상 안전보건조치 위반과 중처법상 의무 위반 수사에서 검찰/노동부의 중점 확인 사항은?", ("제38조", "예산", "실질적으로")),
    ("도급·용역·위탁 시 수급인의 안전보건 이행능력 평가 기준은 어떻게 마련해야 하는가?", ("제4조제9호", "반기 1회")),
    ("안전보건 예산 편성과 집행 기준은 어떻게 세워야 인정받는가?", ("제4조제4호", "실제 집행")),
    ("종사자 의견 수렴 절차 및 재해예방 개선방안 마련 주기는?", ("제4조제7호", "반기 1회")),
    ("수사 기관(노동부·검찰)에서 요구하는 핵심 증빙 서류에는 어떤 것이 있는가?", ("5년간 보관", "위험성평가")),
    ("기업이 지금 당장 실행해야 할 핵심 안전보건 체크리스트 3가지?", ("1.", "2.", "3.", "제36조")),
]


def test_corporate_questions_are_independent_verified_routes() -> None:
    intents = []
    answers = []
    for index, (question, expected) in enumerate(CASES, 1):
        intent = classify_general_law_question(question)
        answer = direct_general_law_answer(intent or "")
        sources = general_law_sources(intent or "")

        assert intent == f"corporate_qa_{index:02d}"
        assert answer
        assert all(value in answer for value in expected)
        assert sources
        assert validate_answer_citations(answer, sources)["status"] != "fail"
        intents.append(intent)
        answers.append(answer)

    assert len(set(intents)) == 20
    assert len(set(answers)) == 20


def test_corporate_general_answers_do_not_leak_accident_routes() -> None:
    forbidden = ("스파일럿건설", "비계 작업", "굴착면", "별표 5 제19호", "별표 5 제23호")
    for question, _ in CASES:
        intent = classify_general_law_question(question)
        answer = direct_general_law_answer(intent or "") or ""
        assert not any(value in answer for value in forbidden)
