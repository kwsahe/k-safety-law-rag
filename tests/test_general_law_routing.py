import pytest

from rag.general_law_routing import classify_general_law_question, direct_general_law_answer


@pytest.mark.parametrize(
    ("question", "expected", "forbidden"),
    [
        (
            "산업안전보건법이 제정된 궁극적인 목적은 무엇인가?",
            ("산업안전보건법 제1조", "노무를 제공하는 사람", "산업재해를 예방"),
            ("산업안전보건기준에 관한 규칙 제1조가 근거",),
        ),
        (
            "사업주와 근로자는 산업재해 예방을 위해 각각 어떤 기본 의무를 가지는가?",
            ("산업안전보건법 제5조", "산업안전보건법 제6조", "제51조", "제57조"),
            ("근로자의 기본 의무는 시행규칙 별표 5",),
        ),
        (
            "사업장에서 안전 및 보건에 관한 업무를 총괄·관리하는 사람은 누구인가?",
            ("안전보건관리책임자", "산업안전보건법 제15조", "산업안전보건법 제62조", "도급"),
            ("중대재해처벌법 제4조",),
        ),
        (
            "사업주가 근로자에게 정기적으로 실시해야 하는 대표적인 교육은 무엇인가?",
            ("매반기 6시간 이상", "매반기 12시간 이상", "시행규칙 별표 4", "시행규칙 별표 5"),
            ("판매업무에 직접 종사하는 근로자: 매반기 3시간", "별표 4 제19호"),
        ),
        (
            "산업안전보건법에서 규정하는 '위험성평가'란 무엇인가?",
            ("산업안전보건법 제36조", "유해ㆍ위험요인", "허용 가능한 범위", "위험 감소조치"),
            ("시행규칙 p.109",),
        ),
    ],
)
def test_general_law_answers(question: str, expected: tuple[str, ...], forbidden: tuple[str, ...]) -> None:
    intent = classify_general_law_question(question)
    assert intent

    answer = direct_general_law_answer(intent)
    assert answer
    assert all(item in answer for item in expected)
    assert all(item not in answer for item in forbidden)
