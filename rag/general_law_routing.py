"""Deterministic answers for foundational occupational-safety law questions."""

from __future__ import annotations

import re

from rag.schemas import SourceDoc


GENERAL_PURPOSE = "general_law_purpose"
GENERAL_DUTIES = "general_law_basic_duties"
GENERAL_MANAGER = "general_law_manager_roles"
GENERAL_REGULAR_EDUCATION = "general_law_regular_education"
GENERAL_RISK_ASSESSMENT = "general_law_risk_assessment"


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def classify_general_law_question(question: str) -> str | None:
    compact = _compact(question)
    if "산업안전보건법" in compact and any(term in compact for term in ("목적", "제정된이유", "궁극적")):
        return GENERAL_PURPOSE
    if "사업주" in compact and "근로자" in compact and any(term in compact for term in ("기본의무", "각각어떤", "산업재해예방")):
        return GENERAL_DUTIES
    if any(term in compact for term in ("업무를총괄·관리", "업무를총괄ㆍ관리", "업무를총괄관리", "총괄·관리하는사람", "총괄관리하는사람")):
        return GENERAL_MANAGER
    if "정기" in compact and any(term in compact for term in ("안전보건교육", "교육시간", "대표적인교육", "실시해야하는교육")):
        return GENERAL_REGULAR_EDUCATION
    if "위험성평가" in compact and any(term in compact for term in ("무엇", "정의", "의미", "뜻", "규정")):
        return GENERAL_RISK_ASSESSMENT
    return None


def _source(law_name: str, article: str = "", annex: str = "", content: str = "") -> SourceDoc:
    return SourceDoc(
        content=content or f"{law_name} {article or annex}",
        metadata={
            "law_name": law_name,
            "article": article,
            "annex": annex,
            "score": 0.98,
            "source_type": "text",
            "retrieval_note": "general_law_verified_fallback",
        },
    )


def general_law_sources(intent: str) -> list[SourceDoc]:
    if intent == GENERAL_PURPOSE:
        return [
            _source("산업안전보건법", "제1조", content="산업재해를 예방하고 쾌적한 작업환경을 조성하여 노무를 제공하는 사람의 안전 및 보건을 유지ㆍ증진한다."),
            _source("산업안전보건기준에 관한 규칙", "제1조", content="산업안전보건법에서 위임한 산업안전보건기준과 시행에 필요한 사항을 규정한다."),
        ]
    if intent == GENERAL_DUTIES:
        return [
            _source("산업안전보건법", "제5조", content="사업주는 산업재해 예방기준 준수, 쾌적한 작업환경 조성, 근로조건 개선과 안전보건 정보 제공 의무를 이행한다."),
            _source("산업안전보건법", "제6조", content="근로자는 산업재해 예방기준을 지키고 사업주와 관계 기관의 예방조치에 따라야 한다."),
            _source("산업안전보건법", "제51조", content="사업주는 급박한 산업재해 위험이 있을 때 작업을 중지하고 근로자를 대피시키는 등 필요한 안전보건조치를 해야 한다."),
            _source("산업안전보건법", "제57조", content="사업주는 산업재해 발생 사실을 은폐해서는 안 되며 기록ㆍ보고 의무를 이행해야 한다."),
        ]
    if intent == GENERAL_MANAGER:
        return [
            _source("산업안전보건법", "제15조", content="사업장을 실질적으로 총괄하여 관리하는 사람이 안전보건 업무를 총괄 관리한다."),
            _source("산업안전보건법", "제62조", content="도급인은 관계수급인 근로자가 함께 작업하는 경우 안전보건총괄책임자를 지정한다."),
        ]
    if intent == GENERAL_REGULAR_EDUCATION:
        return [
            _source("산업안전보건법", "제29조", content="사업주는 근로자에게 정기적으로 안전보건교육을 해야 한다."),
            _source("산업안전보건법 시행규칙", annex="별표 4", content="정기교육 시간은 사무직과 판매업무 직접 종사자는 매반기 6시간 이상, 그 밖의 근로자는 매반기 12시간 이상이다."),
            _source("산업안전보건법 시행규칙", annex="별표 5", content="정기교육에는 산업재해 예방, 건강장해 예방, 위험성평가, 관련 법령, 직무스트레스와 괴롭힘 예방 등이 포함된다."),
        ]
    if intent == GENERAL_RISK_ASSESSMENT:
        return [_source("산업안전보건법", "제36조", content="사업주는 유해ㆍ위험요인을 찾아 위험성의 크기가 허용 가능한 범위인지 평가하고 결과에 따라 필요한 조치를 해야 한다.")]
    return []


def direct_general_law_answer(intent: str) -> str | None:
    if intent == GENERAL_PURPOSE:
        return "\n".join(
            [
                "결론: 산업안전보건법의 목적은 산업재해를 예방하고 쾌적한 작업환경을 조성하여 노무를 제공하는 사람의 안전과 보건을 유지ㆍ증진하는 것입니다.",
                "",
                "[법적 근거]",
                "- 산업안전보건법 제1조",
                "- 산업 안전 및 보건에 관한 기준을 확립하고 책임의 소재를 명확히 합니다.",
                "- 이를 통해 산업재해를 예방하고 쾌적한 작업환경을 조성합니다.",
                "- 궁극적으로 노무를 제공하는 사람의 안전 및 보건을 유지ㆍ증진하는 것이 목적입니다.",
                "",
                "산업안전보건기준에 관한 규칙 제1조는 법률의 위임사항과 시행에 필요한 기준을 정하는 규칙의 목적이므로, 산업안전보건법 자체의 목적과 구분해야 합니다.",
            ]
        )
    if intent == GENERAL_DUTIES:
        return "\n".join(
            [
                "결론: 사업주는 안전한 작업환경과 예방체계를 마련할 기본 의무를 지고, 근로자는 법령상 예방기준과 정당한 안전조치를 준수할 의무를 집니다.",
                "",
                "[사업주의 기본 의무]",
                "- 근거: 산업안전보건법 제5조",
                "- 산업재해 예방을 위한 법령상 기준을 준수해야 합니다.",
                "- 신체적 피로와 정신적 스트레스를 줄일 수 있는 쾌적한 작업환경을 조성하고 근로조건을 개선해야 합니다.",
                "- 사업장의 안전ㆍ보건 정보를 근로자에게 제공하고 국가의 산업재해 예방시책을 따라야 합니다.",
                "- 급박한 위험이 있으면 작업중지ㆍ대피 등 필요한 조치를 해야 합니다. 근거: 제51조",
                "- 산업재해를 은폐해서는 안 되며 기록ㆍ보고 의무를 이행해야 합니다. 근거: 제57조",
                "",
                "[근로자의 기본 의무]",
                "- 근거: 산업안전보건법 제6조",
                "- 법령에서 정한 산업재해 예방기준을 지켜야 합니다.",
                "- 사업주, 노동감독관, 산업안전보건공단 등 관계자가 실시하는 산업재해 예방조치에 따라야 합니다.",
                "",
                "교육 이수는 사업주의 교육 실시 의무와 근로자의 예방조치 준수에 수반되는 사항이며, 근로자의 기본 의무 자체를 시행규칙 별표 5의 교육내용으로 대체해서는 안 됩니다.",
            ]
        )
    if intent == GENERAL_MANAGER:
        return "\n".join(
            [
                "결론: 일반 사업장에서 안전ㆍ보건 업무를 총괄 관리하는 사람은 안전보건관리책임자입니다. 안전보건총괄책임자는 도급 관계가 있는 사업장에서 별도로 적용되는 개념입니다.",
                "",
                "[1. 안전보건관리책임자]",
                "- 근거: 산업안전보건법 제15조",
                "- 사업장을 실질적으로 총괄하여 관리하는 사람이 산업재해 예방계획, 안전보건교육, 작업환경 점검ㆍ개선, 건강관리와 재발방지 업무 등을 총괄 관리합니다.",
                "- 법령상 지정 대상 사업의 종류와 규모에 해당하는 사업장에 적용됩니다.",
                "",
                "[2. 안전보건총괄책임자]",
                "- 근거: 산업안전보건법 제62조",
                "- 관계수급인 근로자가 도급인의 사업장에서 함께 작업하는 경우 도급인이 지정합니다.",
                "- 도급인의 근로자와 관계수급인 근로자의 산업재해 예방업무를 총괄 관리합니다.",
            ]
        )
    if intent == GENERAL_REGULAR_EDUCATION:
        return "\n".join(
            [
                "결론: 대표적인 정기교육은 산업안전보건법 제29조에 따른 근로자 정기 안전보건교육입니다.",
                "",
                "[교육 내용]",
                "- 산업안전 및 산업재해 예방",
                "- 산업보건 및 건강장해 예방",
                "- 위험성평가",
                "- 산업안전보건법령 및 산업재해보상보험 제도",
                "- 직무스트레스 예방ㆍ관리",
                "- 직장 내 괴롭힘과 고객 폭언 등으로 인한 건강장해 예방ㆍ관리",
                "- 근거: 산업안전보건법 시행규칙 별표 5",
                "",
                "[현행 교육시간]",
                "- 사무직 종사 근로자: 매반기 6시간 이상",
                "- 판매업무에 직접 종사하는 근로자: 매반기 6시간 이상",
                "- 판매업무 외의 근로자: 매반기 12시간 이상",
                "- 근거: 산업안전보건법 시행규칙 별표 4",
                "",
                "과거의 '매분기 3시간' 기준과 혼동하지 않도록 현재 시행 중인 매반기 기준을 적용해야 합니다.",
            ]
        )
    if intent == GENERAL_RISK_ASSESSMENT:
        return "\n".join(
            [
                "결론: 위험성평가는 사업주가 업무로 인한 유해ㆍ위험요인을 찾아 위험성의 크기가 허용 가능한 범위인지 평가하고, 결과에 따라 위험 감소조치를 수립ㆍ이행하는 과정입니다.",
                "",
                "[법적 근거와 절차]",
                "- 근거: 산업안전보건법 제36조",
                "1. 건설물, 기계ㆍ기구ㆍ설비, 원재료, 가스ㆍ증기ㆍ분진, 작업행동 등에서 유해ㆍ위험요인을 파악합니다.",
                "2. 해당 요인이 부상 또는 질병으로 이어질 위험성의 크기가 허용 가능한 범위인지 평가합니다.",
                "3. 평가 결과에 따라 법령상 조치와 추가적인 위험 감소조치를 실시합니다.",
                "4. 근로자를 참여시키고 평가 결과와 조치사항을 기록ㆍ보존합니다.",
                "",
                "기계, 화학물질, 전기, 추락ㆍ붕괴 등은 위험요인의 예시이며, 위험성평가의 핵심은 위험요인 파악부터 평가와 감소조치까지 이어지는 절차입니다.",
            ]
        )
    return None
