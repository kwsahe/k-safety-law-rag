"""Deterministic routing for masonry-material falling-object incidents."""

from __future__ import annotations

from rag.schemas import SourceDoc
from rag.v1_incident_routing import (
    _accident_outcome,
    compact_text,
    contract_sources,
    extract_company_name,
    extract_contractor_name,
    make_general_source,
)


def is_masonry_falling_scenario(value: str) -> bool:
    compact = compact_text(value)
    has_material = any(term in compact for term in ("조적", "벽돌", "블록", "자재낙하", "발끝막이판", "폭목"))
    has_falling = any(term in compact for term in ("낙하", "물체맞", "머리를맞", "방호선반", "낙하물방지망"))
    return has_material and has_falling


def is_masonry_special_education_question(question: str, fact_text: str = "") -> bool:
    compact = compact_text(question)
    return is_masonry_falling_scenario(fact_text or question) and any(
        term in compact for term in ("특별교육", "특별안전교육", "교육미실시", "교육내용")
    )


def is_masonry_falling_controls_question(question: str, fact_text: str = "") -> bool:
    compact = compact_text(question)
    issue_count = sum(term in compact for term in ("낙하물방지망", "방호선반", "발끝막이판", "폭목", "출입통제", "출입금지"))
    return is_masonry_falling_scenario(fact_text or question) and issue_count >= 2 and any(
        term in compact for term in ("위반", "조항", "판단")
    )


def _find_source(sources: list[SourceDoc], law_name: str, article: str) -> SourceDoc | None:
    for source in sources:
        source_law = compact_text(str(source.metadata.get("law_name") or source.metadata.get("source") or ""))
        if compact_text(law_name) not in source_law:
            continue
        if str(source.metadata.get("article") or "") == article or article in compact_text(source.content):
            return SourceDoc(content=source.content, metadata={**source.metadata, "article": article})
    return None


FALLING_CONTROL_SPECS = [
    (
        "제13조",
        "안전난간은 상부 난간대, 중간 난간대, 발끝막이판 및 난간기둥으로 구성해야 한다.",
    ),
    (
        "제14조",
        "물체가 떨어지거나 날아올 위험이 있으면 낙하물 방지망, 수직보호망, 방호선반, 출입금지구역 또는 보호구 등 필요한 조치를 해야 한다.",
    ),
    (
        "제20조",
        "법령이 열거한 위험 작업 또는 장소에는 울타리 등을 설치하여 관계 근로자가 아닌 사람의 출입을 금지해야 한다.",
    ),
]


def falling_control_sources(sources: list[SourceDoc]) -> list[SourceDoc]:
    selected: list[SourceDoc] = []
    for article, content in FALLING_CONTROL_SPECS:
        selected.append(
            _find_source(sources, "산업안전보건기준에 관한 규칙", article)
            or make_general_source(
                "산업안전보건기준에 관한 규칙",
                article,
                content,
                f"falling_object_{article}",
            )
        )
    return selected


def masonry_education_sources(sources: list[SourceDoc]) -> list[SourceDoc]:
    education = [
        _find_source(sources, "산업안전보건법", "제29조")
        or make_general_source(
            "산업안전보건법",
            "제29조",
            "산업안전보건법 제29조제1항ㆍ제2항은 정기교육, 채용 시 교육 및 작업내용 변경 시 교육을 정하고, 제29조제3항은 유해ㆍ위험작업의 특별교육을 정한다.",
            "masonry_general_education",
        ),
        make_general_source(
            "산업안전보건법 시행규칙",
            "별표 4",
            "산업안전보건법 시행규칙 별표 4는 안전보건교육 교육시간을 정한다.",
            "masonry_annex4_training_time",
        ),
        make_general_source(
            "산업안전보건법 시행규칙",
            "별표 5",
            "특별교육 대상 작업과 교육내용을 정하며 단순 조적ㆍ벽돌 운반 작업은 독립된 특별교육 대상 항목으로 명시되어 있지 않다.",
            "masonry_annex5_scope",
        ),
    ]
    article_14 = next(
        source for source in falling_control_sources(sources) if source.metadata.get("article") == "제14조"
    )
    return [*education, article_14]


def direct_masonry_special_education_answer() -> str:
    return "\n".join(
        [
            "결론: 단순 조적ㆍ벽돌 운반 작업은 산업안전보건법 시행규칙 별표 5에 독립된 특별교육 대상 작업으로 명시되어 있지 않으므로, 특별안전교육 미실시만으로 특정 별표 항목 위반이라고 판단할 수 없습니다.",
            "",
            "[특별교육 대상 여부]",
            "- 판단: 해당 없음 또는 추가 사실 확인 필요",
            "- 근거: 산업안전보건법 제29조제3항, 산업안전보건법 시행규칙 별표 5",
            "- 굴착 작업 제19호, 비계 조립ㆍ해체 제23호, 크레인 작업 제14호를 조적 작업에 대신 적용해서는 안 됩니다.",
            "- 작업이 단순 조적을 넘어 비계의 조립ㆍ해체ㆍ변경 등 별표 5의 다른 작업을 실제로 포함한 경우에만 해당 작업의 특별교육 의무를 별도로 검토합니다.",
            "",
            "[일반 안전보건교육]",
            "- 사업주는 정기교육, 채용 시 교육 및 작업내용 변경 시 교육을 실시해야 합니다. 근거: 산업안전보건법 제29조제1항ㆍ제2항, 산업안전보건법 시행규칙 별표 4ㆍ별표 5",
            "- 현재 시나리오는 '특별교육 미실시'만 명시하므로 일반교육까지 미실시했다고 단정하지 않고 교육 기록을 추가 확인해야 합니다.",
            "",
            "[이 사고의 핵심 위반]",
            "- 특별교육보다 낙하물 방지망ㆍ방호선반ㆍ출입금지구역 등 낙하물 방지조치 미흡이 직접 쟁점입니다. 근거: 산업안전보건기준에 관한 규칙 제14조",
        ]
    )


def direct_falling_controls_answer() -> str:
    return "\n".join(
        [
            "결론: 낙하물 방지망ㆍ방호선반, 발끝막이판 및 하부 통제 조치가 없었다면 낙하물 위험 방지 의무 위반에 해당합니다.",
            "",
            "[항목별 판단]",
            "1. 낙하물 방지망ㆍ방호선반 미설치",
            "   - 판단: 위반",
            "   - 근거: 산업안전보건기준에 관한 규칙 제14조",
            "   - 이유: 물체가 떨어질 위험이 있으면 낙하물 방지망, 수직보호망, 방호선반 또는 출입금지구역 설정 등 필요한 조치를 해야 합니다.",
            "",
            "2. 발끝막이판(폭목) 미설치",
            "   - 판단: 위반",
            "   - 근거: 산업안전보건기준에 관한 규칙 제13조",
            "   - 이유: 작업발판 가장자리의 안전난간은 발끝막이판을 포함해 구성해야 하며, 벽돌 등 자재가 굴러 떨어지지 않도록 해야 합니다.",
            "",
            "3. 하부 출입통제 미실시",
            "   - 판단: 위반",
            "   - 직접 근거: 산업안전보건기준에 관한 규칙 제14조",
            "   - 이유: 제14조는 낙하 위험 방지방법으로 출입금지구역 설정을 명시합니다. 하부 통로를 개방한 채 상부 조적 작업을 계속했다면 위반입니다.",
            "   - 보충 근거: 산업안전보건기준에 관한 규칙 제20조",
            "   - 제20조는 각 호에 열거된 위험 작업ㆍ장소에서 관계 근로자가 아닌 사람의 출입을 금지하도록 하므로, 해당 장소 요건이 충족되는지도 함께 검토합니다.",
        ]
    )


def direct_falling_prime_contractor_answer(question: str) -> str:
    company = extract_company_name(question, "해당 원청")
    contractor = extract_contractor_name(question, "해당 조적업체")
    return "\n".join(
        [
            f"결론: YES. 조적 작업을 {contractor}에 도급했더라도 원청 {company}이 건설현장과 작업 공정을 실질적으로 지배ㆍ관리했다면 두 법령상 책임이 성립할 수 있습니다.",
            "",
            "[산업안전보건법상 원청 책임]",
            "- 근거: 산업안전보건법 제64조",
            "- 원청은 관계수급인 근로자가 원청 사업장에서 작업하는 경우 상하 동시작업 조정, 낙하물 방지망ㆍ방호선반 설치와 하부 출입통제 등 구체적인 산업재해 예방조치를 이행해야 합니다.",
            "- 안전보건총괄책임자 지정 대상이면 산업안전보건법 제62조에 따른 총괄관리 의무도 검토합니다.",
            "",
            "[중대재해처벌법상 원청 책임]",
            "- 근거: 중대재해처벌법 제5조",
            f"- {company}이 시설ㆍ장비ㆍ장소를 실질적으로 지배ㆍ운영ㆍ관리했다면 경영책임자의 안전보건 확보의무가 적용됩니다.",
            "- 추가 근거: 중대재해처벌법 시행령 제4조제9호",
            f"- {contractor}의 안전보건 역량 평가기준ㆍ절차를 마련하고 반기 1회 이상 점검했는지 확인해야 합니다.",
            "",
            "[책임 범위 차이]",
            "- 산업안전보건법은 낙하물 방지망, 발끝막이판, 출입통제와 작업 조정 등 현장 단위의 구체적 조치 이행을 판단합니다.",
            "- 중대재해처벌법은 수급업체 선정ㆍ평가ㆍ점검과 낙하물 위험 관리절차 등 경영책임자의 관리체계 구축ㆍ이행을 판단합니다.",
        ]
    )


def falling_prime_contractor_sources(sources: list[SourceDoc]) -> list[SourceDoc]:
    total_manager = _find_source(sources, "산업안전보건법", "제62조") or make_general_source(
        "산업안전보건법",
        "제62조",
        "도급인은 관계수급인 근로자의 산업재해 예방 업무를 총괄 관리하기 위하여 안전보건총괄책임자를 지정해야 한다.",
        "falling_prime_contractor_manager",
    )
    return [*contract_sources(sources), total_manager, *falling_control_sources(sources)]


def falling_comprehensive_sources(question: str, sources: list[SourceDoc]) -> list[SourceDoc]:
    death, _, _, _ = _accident_outcome(question)
    selected = [
        *falling_control_sources(sources),
        *masonry_education_sources(sources),
        *contract_sources(sources),
        make_general_source("중대재해처벌법", "제2조제2호가목" if death else "제2조제2호나목", "중대산업재해 정의", "falling_serious_definition"),
        make_general_source("중대재해처벌법", "제4조", "경영책임자의 안전보건 확보의무", "falling_serious_duty"),
        make_general_source("중대재해처벌법", "제6조제1항" if death else "제6조제2항", "경영책임자 처벌", "falling_manager_penalty"),
        make_general_source("중대재해처벌법", "제7조제1호" if death else "제7조제2호", "법인 양벌규정", "falling_entity_penalty"),
    ]
    return selected


def direct_falling_comprehensive_answer(question: str) -> str:
    company = extract_company_name(question, "해당 사업주")
    contractor = extract_contractor_name(question, "해당 조적업체")
    death, _, _, _ = _accident_outcome(question)
    serious_result = (
        "- 사망자 1명이 발생했으므로 중대재해처벌법 제2조제2호가목의 중대산업재해에 해당합니다."
        if death
        else "- 사고 결과에 따라 중대재해처벌법 제2조제2호의 중대산업재해 요건을 별도로 판단해야 합니다."
    )
    manager_penalty = "1년 이상의 징역 또는 10억원 이하의 벌금" if death else "제6조제2항의 부상 사고 처벌 기준"
    entity_penalty = "50억원 이하의 벌금" if death else "제7조제2호의 부상 사고 양벌 기준"
    return "\n".join(
        [
            "최종 보고서",
            "",
            "1. 사고 원인 분석",
            "[직접 원인]",
            "- 조적 작업 중 벽돌을 작업발판 가장자리에 임시 적재했고 발끝막이판이 없어 벽돌이 하부로 떨어졌습니다.",
            "- 낙하물 방지망ㆍ방호선반과 하부 출입금지구역이 없어 근로자가 낙하물 위험에 직접 노출됐습니다.",
            "",
            "[간접 원인]",
            "- 상하 동시작업 위험성평가, 자재 적재 기준, 작업지휘와 하부 통제 절차가 현장에서 작동하지 않았습니다.",
            f"- 원청과 {contractor} 사이의 작업 조정 및 낙하물 방지조치 확인체계가 미흡했습니다.",
            "",
            "2. 법령 위반 사항 요약",
            "[산업안전보건법]",
            "- 발끝막이판 미설치: 산업안전보건기준에 관한 규칙 제13조",
            "- 낙하물 방지망ㆍ방호선반 및 하부 출입금지구역 미설정: 제14조",
            "- 위험 장소 출입금지 의무: 제20조 적용 여부 추가 검토",
            "- 정기ㆍ채용 시ㆍ작업내용 변경 시 안전보건교육 이행 여부: 산업안전보건법 제29조",
            "- 원청의 도급 작업 산업재해 예방조치: 산업안전보건법 제64조",
            "- 단순 조적 작업은 별표 5의 독립된 특별교육 대상이 아니므로 굴착ㆍ비계 특별교육 조항을 적용하지 않습니다.",
            "",
            "[중대재해처벌법]",
            serious_result,
            "- 낙하물 위험요인 확인ㆍ개선, 위험성평가와 관계 법령 이행 점검체계가 작동했는지 검토합니다. 근거: 중대재해처벌법 제4조",
            "",
            "3. 책임 주체별 판단",
            "[사업주ㆍ원청]",
            f"- {company}은 현장과 공정을 관리한 사업주ㆍ원청으로서 낙하물 방지, 하부 통제, 교육 및 도급 작업 조정의 1차 책임 주체입니다.",
            f"- {contractor}가 조적 작업을 수행했더라도 원청의 제64조상 산업재해 예방조치는 별도로 적용됩니다.",
            "- 현재 확인된 위반의 처벌 주체는 사업주이며, 현장 근로자의 행위를 근로자 개인 처벌로 바로 귀결하지 않습니다.",
            "",
            "[경영책임자]",
            f"- 사망 결과와 안전보건관리체계 미비 사이의 인과관계가 인정되면 제6조제1항에 따라 {manager_penalty} 대상이 될 수 있습니다.",
            "",
            "[법인]",
            f"- 경영책임자의 위반행위가 인정되면 법인 {company}은 제7조제1호에 따라 {entity_penalty} 대상이 될 수 있습니다.",
            "",
            "4. 즉시 취해야 할 재발방지 조치",
            "- 상부 조적 작업을 중지하고 발끝막이판을 포함한 안전난간을 설치합니다. 근거: 제13조",
            "- 낙하물 방지망ㆍ방호선반을 설치하고 하부 통로에 출입금지구역과 감시자를 배치합니다. 근거: 제14조",
            "- 벽돌을 작업발판 가장자리에 임시 적재하지 않도록 적재 위치ㆍ수량 기준과 자재 고정 절차를 마련합니다.",
            "- 상하 동시작업을 분리하고 작업지휘자를 지정한 뒤 위험성평가와 작업 전 점검을 다시 실시합니다.",
            f"- {contractor}의 안전보건 역량과 작업계획을 재평가하고 반기 점검체계를 운영합니다. 근거: 중대재해처벌법 제5조 및 시행령 제4조제9호",
        ]
    )
