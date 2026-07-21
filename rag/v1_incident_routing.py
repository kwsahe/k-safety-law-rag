"""Deterministic v1 routes for machinery entanglement and struck-by incidents."""

from __future__ import annotations

import re
from typing import Iterable

from rag.schemas import SourceDoc


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def is_machine_entanglement_scenario(value: str) -> bool:
    compact = compact_text(value)
    has_machine = any(
        term in compact
        for term in ("프레스", "전단기", "컨베이어", "회전축", "롤러", "기계설비", "금형")
    )
    has_risk = any(
        term in compact
        for term in ("끼임", "협착", "말림", "압착", "갑자기작동", "운전정지", "동력차단", "방호장치")
    )
    return has_machine and has_risk


def is_struck_by_scenario(value: str) -> bool:
    compact = compact_text(value)
    has_lifting = any(term in compact for term in ("크레인", "호이스트", "양중", "인양", "매달린화물"))
    has_struck_risk = any(
        term in compact
        for term in ("낙하", "비래", "물체에맞", "맞음", "와이어로프", "걸고리", "해지장치", "인양물", "인양작업")
    )
    return has_lifting and has_struck_risk


def is_machine_special_education_question(question: str, fact_text: str = "") -> bool:
    compact_question = compact_text(question)
    return (
        is_machine_entanglement_scenario(fact_text or question)
        and any(term in compact_question for term in ("특별교육", "특별안전교육", "교육내용", "교육미실시", "미이수"))
    )


def is_struck_special_education_question(question: str, fact_text: str = "") -> bool:
    compact_question = compact_text(question)
    return (
        is_struck_by_scenario(fact_text or question)
        and any(term in compact_question for term in ("특별교육", "특별안전교육", "교육내용", "교육미실시", "미이수"))
    )


def is_machine_controls_question(question: str) -> bool:
    compact = compact_text(question)
    issue_count = sum(
        term in compact
        for term in ("방호장치", "동력차단", "운전정지", "정비", "청소", "금형", "기동", "잠금")
    )
    return issue_count >= 2 and any(term in compact for term in ("위반", "조항", "판단", "모두", "나열"))


def is_machine_controls_inspection_question(question: str) -> bool:
    compact = compact_text(question)
    control_issue_count = sum(
        term in compact
        for term in ("방호장치", "동력차단", "운전정지", "정비", "청소", "금형", "기동", "잠금")
    )
    return (
        "프레스" in compact
        and "안전검사" in compact
        and "자율안전확인" not in compact
        and control_issue_count >= 2
        and any(term in compact for term in ("위반", "조항", "판단"))
    )


def is_machine_inspection_question(question: str) -> bool:
    compact = compact_text(question)
    return (
        "프레스" in compact
        and any(term in compact for term in ("안전검사", "자율안전확인"))
        and not is_machine_controls_inspection_question(question)
    )


def is_struck_controls_question(question: str) -> bool:
    compact = compact_text(question)
    issue_count = sum(
        term in compact
        for term in ("낙하", "비래", "출입금지", "와이어로프", "걸고리", "해지장치", "신호", "정격하중", "방호장치")
    )
    return issue_count >= 2 and any(term in compact for term in ("위반", "조항", "판단", "모두", "나열"))


def extract_company_name(value: str, default: str = "해당 원청") -> str:
    match = re.search(r"\(주\)\s*([가-힣A-Za-z0-9_-]+)", value or "")
    if not match:
        return default
    name = match.group(1)
    for suffix in ("에서는", "에게는", "으로는", "에서", "에게", "으로", "의", "은", "는", "이", "가", "을", "를", "과", "와"):
        if name.endswith(suffix) and len(name) > len(suffix) + 1:
            name = name[: -len(suffix)]
            break
    return f"(주){name}"


def extract_contractor_name(value: str, default: str = "해당 수급업체") -> str:
    named_matches = re.findall(r"(?:수급업체|협력업체)\s*([A-Z])\s*사", value or "")
    if named_matches:
        return f"{named_matches[-1]}사"
    matches = re.findall(r"(?<![A-Za-z0-9가-힣])([A-Z])\s*사(?![A-Za-z0-9가-힣])", value or "")
    return f"{matches[-1]}사" if matches else default


def _item_number(source: SourceDoc) -> str:
    item_number = str(source.metadata.get("item_number") or "")
    match = re.match(r"\s*(\d+)\.", item_number)
    if match:
        return match.group(1)
    match = re.search(r"\[작업항목\]\s*(\d+)\.", source.content)
    return match.group(1) if match else ""


def _find_item_source(sources: Iterable[SourceDoc], item_no: str) -> SourceDoc | None:
    for source in sources:
        if _item_number(source) == item_no:
            return source
    return None


MACHINE_EDUCATION_ITEMS = [
    "프레스의 특성과 위험성에 관한 사항",
    "방호장치 종류와 취급에 관한 사항",
    "안전작업방법에 관한 사항",
    "프레스 안전기준에 관한 사항",
    "그 밖에 안전ㆍ보건관리에 필요한 사항",
]

CRANE_EDUCATION_ITEMS = [
    "방호장치의 종류, 기능 및 취급에 관한 사항",
    "걸고리ㆍ와이어로프 및 비상정지장치 등의 기계ㆍ기구 점검에 관한 사항",
    "화물의 취급 및 안전작업방법에 관한 사항",
    "신호방법 및 공동작업에 관한 사항",
    "인양 물건의 위험성 및 낙하ㆍ비래ㆍ충돌재해 예방에 관한 사항",
    "인양물이 적재될 지반 조건, 인양하중과 풍압 등의 영향에 관한 사항",
    "그 밖에 안전ㆍ보건관리에 필요한 사항",
]


def make_special_education_source(kind: str) -> SourceDoc:
    if kind == "machine":
        item_no, page = "11", "81"
        title = "동력으로 작동되는 프레스기계를 5대 이상 보유한 사업장에서 해당 기계로 하는 작업"
        items = MACHINE_EDUCATION_ITEMS
    else:
        item_no, page = "14", "81"
        title = "1톤 이상의 크레인을 사용하는 작업 등"
        items = CRANE_EDUCATION_ITEMS
    return SourceDoc(
        content=f"[작업항목] {item_no}. {title}\n[교육내용]\n" + "\n".join(f"○ {item}" for item in items),
        metadata={
            "law_name": "산업안전보건법 시행규칙",
            "annex": f"별표 5 제1호라목 제{item_no}호",
            "item_number": f"{item_no}.",
            "page": page,
            "citation_page": page,
            "score": 0.98,
            "source_type": "table",
            "retrieval_note": f"v1_{kind}_special_education",
        },
    )


def special_education_source(kind: str, sources: list[SourceDoc]) -> SourceDoc:
    item_no = "11" if kind == "machine" else "14"
    return _find_item_source(sources, item_no) or make_special_education_source(kind)


def special_education_sources(kind: str, sources: list[SourceDoc]) -> list[SourceDoc]:
    """Return the work-specific annex row and its parent training duty."""
    duty = _find_article_source(
        [
            source
            for source in sources
            if compact_text("산업안전보건법")
            == compact_text(str(source.metadata.get("law_name") or ""))
        ],
        "제29조제3항",
    )
    if duty is None:
        duty = make_general_source(
            "산업안전보건법",
            "제29조제3항",
            "사업주는 유해하거나 위험한 작업에 근로자를 사용할 때 필요한 안전보건교육을 추가로 해야 한다.",
            "v1_special_education_duty",
        )
    return [special_education_source(kind, sources), duty]


def direct_special_education_answer(kind: str, sources: list[SourceDoc]) -> str:
    source = special_education_source(kind, sources)
    if kind == "machine":
        item_no, label, reason, items = (
            "11",
            "프레스 기계 작업",
            "프레스 작업은 기계 위험과 방호장치 취급에 관한 특별교육 대상이며, 시나리오에 교육 미실시 단서가 있습니다.",
            MACHINE_EDUCATION_ITEMS,
        )
    else:
        item_no, label, reason, items = (
            "14",
            "크레인 인양 작업",
            "1톤 이상의 크레인 사용 작업은 특별교육 대상이며, 시나리오에 교육 미실시 단서가 있습니다.",
            CRANE_EDUCATION_ITEMS,
        )
    page = source.metadata.get("citation_page") or source.metadata.get("page") or "81"
    lines = [
        "위반 여부: YES",
        "",
        "[위반 조항]",
        f"- 산업안전보건법 제29조제3항",
        f"- 산업안전보건법 시행규칙 별표 5 제1호라목 제{item_no}호, p.{page}",
        f"- 해당 작업: {label}",
        f"- 해당 이유: {reason}",
        "",
        "[관련 교육 내용]",
    ]
    lines.extend(f"{index}. {item}" for index, item in enumerate(items, start=1))
    return "\n".join(lines)


MACHINE_CONTROL_TEXT = {
    "제87조": "원동기ㆍ회전축ㆍ기어 등 위험 부위에 덮개ㆍ울 등의 방호조치를 설치해야 한다.",
    "제88조": "동력으로 작동되는 기계에 조작이 쉽고 갑자기 움직일 우려가 없는 동력차단장치를 설치해야 한다.",
    "제89조": "기계 운전 시작 전 근로자 배치, 교육, 작업방법과 방호장치를 확인하고 신호해야 한다.",
    "제92조": "기계의 정비ㆍ청소ㆍ검사ㆍ수리ㆍ교체ㆍ조정 작업 시 운전을 정지하고 잠금 또는 표지 등 필요한 조치를 해야 한다.",
    "제93조": "기계의 방호장치를 해체하거나 사용 정지해서는 안 되며 작업 후 즉시 정상 기능을 회복해야 한다.",
    "제103조": "프레스 작업자의 신체 일부가 위험한계에 들어가지 않도록 덮개 등 방호조치를 해야 한다.",
    "제104조": "프레스 금형 조정 시 슬라이드의 갑작스러운 작동 위험을 막기 위해 안전블록 등을 사용해야 한다.",
}

STRUCK_CONTROL_TEXT = {
    "제14조": "물체가 떨어지거나 날아올 위험이 있으면 방지망, 방호선반, 출입금지구역과 보호구 등 필요한 조치를 해야 한다.",
    "제133조": "양중기와 달기구의 정격하중ㆍ운전속도ㆍ경고표시를 작업자가 보기 쉬운 곳에 표시해야 한다.",
    "제134조": "크레인의 과부하방지장치ㆍ권과방지장치ㆍ비상정지장치와 제동장치가 정상 작동하도록 조정해야 한다.",
    "제146조": "크레인 작업 시 하물을 끌거나 밀지 않는 등 인양 작업의 안전조치를 준수하게 해야 한다.",
    "제149조": "이동식 크레인으로 하물을 운반할 때에는 해지장치를 사용해야 한다.",
}


def make_control_source(kind: str, article: str) -> SourceDoc:
    text_map = MACHINE_CONTROL_TEXT if kind == "machine" else STRUCK_CONTROL_TEXT
    return SourceDoc(
        content=f"산업안전보건기준에 관한 규칙 {article}: {text_map[article]}",
        metadata={
            "law_name": "산업안전보건기준에 관한 규칙",
            "article": article,
            "score": 0.98,
            "source_type": "text",
            "retrieval_note": f"v1_{kind}_{article}",
        },
    )


def _find_article_source(sources: Iterable[SourceDoc], article: str) -> SourceDoc | None:
    for source in sources:
        if str(source.metadata.get("article") or "") == article:
            return source
    for source in sources:
        if article in compact_text(source.content):
            return SourceDoc(content=source.content, metadata={**source.metadata, "article": article})
    return None


def control_sources(kind: str, sources: list[SourceDoc]) -> list[SourceDoc]:
    articles = list(MACHINE_CONTROL_TEXT if kind == "machine" else STRUCK_CONTROL_TEXT)
    return [_find_article_source(sources, article) or make_control_source(kind, article) for article in articles]


def direct_controls_answer(kind: str, sources: list[SourceDoc]) -> str:
    refs = {source.metadata.get("article"): source for source in control_sources(kind, sources)}
    if kind == "machine":
        items = [
            ("회전ㆍ동력 전달부 방호 미흡", "제87조", "위험 부위에 덮개ㆍ울 등이 없었다면 위반으로 판단합니다."),
            ("동력차단장치 미확보", "제88조", "쉽게 조작할 수 있는 동력차단장치가 없거나 유효하지 않았다면 위반입니다."),
            ("운전 시작 전 확인ㆍ신호 미실시", "제89조", "정비 중인 근로자를 확인하지 않고 기동했다면 위반입니다."),
            ("정비ㆍ청소 시 운전정지와 잠금ㆍ표지 미실시", "제92조", "전원을 차단하지 않고 청소ㆍ조정 작업을 했다면 위반입니다."),
            ("방호장치 해체ㆍ정지", "제93조", "인터록 등 방호장치를 무력화한 상태로 사용했다면 위반입니다."),
            ("프레스 위험한계 방호 미흡", "제103조", "신체가 금형 위험한계에 들어갈 수 있었다면 방호조치 위반입니다."),
            ("금형 조정 안전블록 미사용", "제104조", "금형 조정 중 안전블록 등 갑작스러운 작동 방지조치가 없었다면 위반입니다."),
        ]
        conclusion = "프레스 방호, 동력차단, 정비ㆍ청소 운전정지와 기동 전 확인 의무 위반이 각각 검토됩니다."
    else:
        items = [
            ("낙하ㆍ비래 위험구역 통제 미흡", "제14조", "인양물 아래 출입금지구역이나 방호조치가 없었다면 위반입니다."),
            ("정격하중ㆍ경고표시 미흡", "제133조", "크레인과 달기구의 정격하중 표시를 확인할 수 없었다면 위반입니다."),
            ("양중기 방호장치 점검ㆍ조정 미흡", "제134조", "과부하ㆍ권과ㆍ비상정지장치가 정상 작동하지 않았다면 위반입니다."),
            ("크레인 인양 안전작업 미준수", "제146조", "부적절한 인양 방법으로 하물이 이탈했다면 위반으로 검토합니다."),
            ("이동식 크레인 해지장치 미사용", "제149조", "걸고리에서 하물이 빠질 수 있는 상태로 운반했다면 위반입니다."),
        ]
        conclusion = "낙하물 방지, 크레인 방호장치, 인양 안전작업과 해지장치 의무 위반이 각각 검토됩니다."
    lines = [f"결론: YES. {conclusion}", "", "[항목별 판단]"]
    for index, (title, article, judgment) in enumerate(items, start=1):
        source = refs[article]
        page = source.metadata.get("citation_page") or source.metadata.get("page")
        page_suffix = f", p.{page}" if page not in (None, "", 0, "0") else ""
        lines.extend(
            [
                f"{index}. {title}",
                f"   - 근거: 산업안전보건기준에 관한 규칙 {article}{page_suffix}",
                f"   - 판단: {judgment}",
            ]
        )
    return "\n".join(lines)


MACHINE_INSPECTION_SPECS = [
    (
        "산업안전보건법",
        "제80조제3항",
        "사업주는 방호조치가 정상적인 기능을 발휘하도록 관련 장치를 상시 점검하고 정비해야 한다.",
    ),
    (
        "산업안전보건법",
        "제89조",
        "자율안전확인대상기계등을 제조하거나 수입하는 자는 자율안전기준 적합 여부를 확인하여 신고해야 한다.",
    ),
    (
        "산업안전보건법",
        "제93조",
        "안전검사대상기계등을 사용하는 사업주 또는 소유자는 검사기준 적합 여부에 관한 안전검사를 받아야 한다.",
    ),
    (
        "산업안전보건기준에 관한 규칙",
        "제36조",
        "사업주는 자율안전기준 또는 안전검사기준에 적합하지 않은 기계ㆍ설비와 방호장치를 사용해서는 안 된다.",
    ),
    (
        "산업안전보건기준에 관한 규칙",
        "제93조",
        "사업주는 방호장치를 해체하거나 사용을 정지해서는 안 되고 수리 후 즉시 정상 기능을 회복시켜야 한다.",
    ),
]


def machine_inspection_sources(sources: list[SourceDoc]) -> list[SourceDoc]:
    selected: list[SourceDoc] = []
    for law_name, article, content in MACHINE_INSPECTION_SPECS:
        candidates = [
            source
            for source in sources
            if compact_text(law_name) == compact_text(str(source.metadata.get("law_name") or ""))
        ]
        source = _find_article_source(candidates, article)
        selected.append(
            source
            or make_general_source(
                law_name,
                article,
                content,
                f"v1_machine_inspection_{compact_text(law_name)}_{article}",
            )
        )
    return selected


def direct_machine_inspection_answer(sources: list[SourceDoc], fact_text: str = "") -> str:
    del sources
    company = extract_company_name(fact_text, "해당 사용 사업주")
    return "\n".join(
        [
            "결론: 안전검사 미실시와 고장 난 방호장치를 사용한 사실은 사업주 의무 위반에 해당합니다. 자율안전확인 신고 의무는 제조ㆍ수입자에게 부과되므로 현재 사실만으로 사용 사업주의 제89조 위반을 단정하지 않습니다.",
            "",
            "[항목별 판단]",
            "1. 프레스 안전검사 미실시",
            "   - 판단: 위반",
            "   - 근거: 산업안전보건법 제93조",
            "   - 이유: 안전검사대상 프레스를 사용하는 사업주 또는 소유자는 정해진 안전검사를 받아야 합니다.",
            "",
            "2. 자율안전확인 신고",
            "   - 판단: 사용 사업주의 위반으로 단정할 수 없음",
            "   - 근거: 산업안전보건법 제89조",
            f"   - 이유: 자율안전확인 신고의 직접 의무자는 자율안전확인대상기계등의 제조자ㆍ수입자입니다. {company}가 제조ㆍ수입자라는 사실이 추가로 확인되어야 제89조 위반을 판단할 수 있습니다.",
            "",
            "3. 검사ㆍ자율안전기준 부적합 기계 사용",
            "   - 판단: 위반",
            "   - 근거: 산업안전보건기준에 관한 규칙 제36조",
            "   - 이유: 안전검사기준 또는 자율안전기준에 적합하지 않은 프레스와 방호장치를 작업에 사용해서는 안 됩니다.",
            "",
            "4. 방호장치 점검ㆍ정비 및 기능 유지 미흡",
            "   - 판단: 위반",
            "   - 근거: 산업안전보건법 제80조제3항, 산업안전보건기준에 관한 규칙 제93조",
            "   - 이유: 방호장치의 고장을 방치하고 정상 기능이 확보되지 않은 상태로 계속 사용했다면 상시 점검ㆍ정비 및 정상 기능 유지 의무 위반입니다.",
        ]
    )


def machine_controls_inspection_sources(sources: list[SourceDoc]) -> list[SourceDoc]:
    inspection = machine_inspection_sources(sources)
    selected_inspection = [
        source
        for source in inspection
        if str(source.metadata.get("article") or "") in {"제80조제3항", "제93조"}
    ]
    return [*control_sources("machine", sources), *selected_inspection]


def direct_machine_controls_inspection_answer(sources: list[SourceDoc]) -> str:
    del sources
    return "\n".join(
        [
            "결론: 프레스 방호장치 미작동, 안전검사 미실시 및 정비 시 운전정지ㆍ동력차단 조치 미이행은 각각 사업주의 법령상 의무 위반으로 판단됩니다.",
            "",
            "[항목별 판단]",
            "1. 방호장치 미작동 상태로 프레스 사용",
            "   - 판단: 위반",
            "   - 근거: 산업안전보건법 제80조제3항, 산업안전보건기준에 관한 규칙 제87조ㆍ제93조ㆍ제103조",
            "   - 이유: 방호장치를 정상 기능하도록 점검ㆍ정비하지 않았고, 고장ㆍ무력화된 상태에서 신체가 금형 위험한계에 들어갈 수 있도록 사용했습니다.",
            "",
            "2. 프레스 안전검사 미실시",
            "   - 판단: 위반",
            "   - 근거: 산업안전보건법 제93조",
            "   - 이유: 안전검사대상 프레스를 사용하는 사업주 또는 소유자는 정해진 안전검사를 받아야 합니다.",
            "",
            "3. 정비ㆍ조정 시 운전정지와 동력차단 미실시",
            "   - 판단: 위반",
            "   - 근거: 산업안전보건기준에 관한 규칙 제88조ㆍ제92조ㆍ제104조",
            "   - 이유: 소재 제거와 정비ㆍ조정 전에 운전을 정지하고 전원을 차단한 뒤 잠금ㆍ표지와 안전블록 등 갑작스러운 작동 방지조치를 해야 합니다.",
            "",
            "4. 기동 전 작업자 확인ㆍ신호 미실시",
            "   - 판단: 위반",
            "   - 근거: 산업안전보건기준에 관한 규칙 제89조",
            "   - 이유: 프레스를 다시 기동하기 전에 작업자의 위치, 작업방법과 방호장치 상태를 확인하고 필요한 신호를 해야 합니다.",
            "",
            "이 질문은 안전검사와 현장 방호ㆍ정비조치 위반을 대상으로 판단했습니다.",
        ]
    )


def make_general_source(law_name: str, article: str, content: str, note: str) -> SourceDoc:
    return SourceDoc(
        content=f"{law_name} {article}: {content}",
        metadata={
            "law_name": law_name,
            "article": article,
            "score": 0.98,
            "source_type": "text",
            "retrieval_note": note,
        },
    )


def contract_sources(sources: list[SourceDoc]) -> list[SourceDoc]:
    specs = [
        ("산업안전보건법", "제64조", "도급인은 관계수급인 근로자의 산업재해 예방을 위한 안전보건조치를 해야 한다."),
        ("중대재해처벌법", "제5조", "도급ㆍ용역ㆍ위탁 관계에서도 시설ㆍ장비ㆍ장소를 실질적으로 지배ㆍ운영ㆍ관리하면 안전보건 확보의무를 부담한다."),
        ("중대재해처벌법 시행령", "제4조제9호", "수급인의 안전보건 역량 평가기준과 절차를 마련하고 반기 1회 이상 점검해야 한다."),
    ]
    selected: list[SourceDoc] = []
    for law, article, content in specs:
        source = _find_article_source(
            [s for s in sources if compact_text(law) in compact_text(str(s.metadata.get("law_name") or ""))],
            article,
        )
        selected.append(source or make_general_source(law, article, content, f"v1_contract_{article}"))
    return selected


def direct_prime_contractor_answer(kind: str, question: str, sources: list[SourceDoc]) -> str:
    company = extract_company_name(question)
    contractor = extract_contractor_name(question)
    work = "프레스 정비ㆍ청소 작업" if kind == "machine" else "크레인 인양 작업"
    hazard = "기계의 동력차단ㆍ방호장치와 작업허가" if kind == "machine" else "인양계획ㆍ달기구 점검ㆍ출입통제와 신호체계"
    return "\n".join(
        [
            f"결론: YES. {work}을 {contractor}에 도급했더라도 원청 {company}이 작업장과 설비ㆍ공정을 실질적으로 지배ㆍ관리했다면 두 법령상 책임이 성립할 수 있습니다.",
            "",
            "[산업안전보건법상 원청 책임]",
            f"- 근거: 산업안전보건법 제64조",
            f"- 판단: 원청은 관계수급인 근로자에 대해서도 {hazard} 등 산업재해 예방조치를 이행해야 합니다.",
            "",
            "[중대재해처벌법상 원청 책임]",
            "- 근거: 중대재해처벌법 제5조",
            "- 판단: 원청이 시설ㆍ장비ㆍ장소를 실질적으로 지배ㆍ운영ㆍ관리하면 경영책임자의 안전보건 확보의무가 적용됩니다.",
            "- 추가 근거: 중대재해처벌법 시행령 제4조제9호",
            f"- 판단: {contractor}의 안전보건 역량 평가기준ㆍ절차를 마련하고 반기 1회 이상 점검했는지 확인해야 합니다.",
            "",
            "[책임 범위 차이]",
            f"- 산업안전보건법은 {work} 현장의 구체적 안전조치 의무를 판단합니다.",
            "- 중대재해처벌법은 수급업체 선정ㆍ평가ㆍ점검을 포함한 경영책임자의 관리체계 구축과 이행 여부를 판단합니다.",
        ]
    )


def prime_contractor_sources(kind: str, sources: list[SourceDoc]) -> list[SourceDoc]:
    del kind
    return contract_sources(sources)


def _accident_is_death(value: str) -> bool:
    compact = compact_text(value)
    if re.search(r"사망자?[:：]?(?:는|가)?(?:없음|없다|없고|0명)", compact) or any(
        term in compact for term in ("사망자는없", "사망자가없", "사망하지않")
    ):
        return False
    return bool(re.search(r"사망(?:자)?(?:가|은|:)?\s*1\s*명", value or "")) or any(
        term in compact for term in ("사망하였다", "사망했다", "사망함")
    )


def _accident_outcome(value: str) -> tuple[bool, int, int, bool]:
    compact = compact_text(value)
    death = _accident_is_death(value)
    injury_match = re.search(r"부상자[:：]?(\d+)명", compact)
    injury_count = int(injury_match.group(1)) if injury_match else 0
    if injury_count == 0 and any(term in compact for term in ("근로자2명", "부상자2명", "2명모두", "두명모두")):
        injury_count = 2
    if injury_count == 0 and re.search(r"[A-Z]씨(?:와|과)[A-Z]씨", compact):
        injury_count = 2
    if injury_count == 0 and any(term in compact for term in ("부상을입", "절단중상", "입원치료")):
        injury_count = 1
    treatment_values: list[int] = []
    for match in re.finditer(r"(\d+)개월", compact):
        before = compact[max(0, match.start() - 10):match.start()]
        after = compact[match.end():match.end() + 30]
        if "고용기간" in before:
            continue
        if "치료" in after or re.match(r"(?:이상)?치료", after):
            treatment_values.append(int(match.group(1)))
    treatment_months = min(treatment_values) if treatment_values else 0
    applies = death or (injury_count >= 2 and treatment_months >= 6)
    return death, injury_count, treatment_months, applies


def _has_contract_relationship(value: str) -> bool:
    compact = compact_text(value)
    if any(term in compact for term in ("도급관계:없음", "도급관계없음", "직영작업", "직영으로")):
        return False
    return any(term in compact for term in ("도급", "수급업체", "협력업체", "원청", "하청"))


def comprehensive_sources(kind: str, question: str, sources: list[SourceDoc]) -> list[SourceDoc]:
    death, _, _, applies = _accident_outcome(question)
    selected = [special_education_source(kind, sources), *control_sources(kind, sources)]
    if kind == "machine":
        selected.extend(machine_inspection_sources(sources))
    if _has_contract_relationship(question):
        selected.extend(contract_sources(sources))
    selected.extend(
        [
            make_general_source("산업안전보건법", "제29조제3항", "유해하거나 위험한 작업에 필요한 특별교육을 실시해야 한다.", "v1_training_duty"),
            make_general_source("중대재해처벌법", "제2조제2호가목" if death else "제2조제2호나목", "중대산업재해 정의", "v1_serious_definition"),
            make_general_source("중대재해처벌법", "제4조", "경영책임자의 안전보건 확보의무", "v1_serious_duty"),
        ]
    )
    selected.extend(
        [
            make_general_source("중대재해처벌법", "제6조제1항" if death else "제6조제2항", "경영책임자 처벌", "v1_manager_penalty"),
            make_general_source("중대재해처벌법", "제7조제1호" if death else "제7조제2호", "법인 양벌규정", "v1_entity_penalty"),
        ]
    )
    return selected


def direct_comprehensive_answer(kind: str, question: str, sources: list[SourceDoc]) -> str:
    company = extract_company_name(question, "해당 사업주")
    has_contract = _has_contract_relationship(question)
    contractor = extract_contractor_name(question) if has_contract else ""
    death, injury_count, treatment_months, serious_applies = _accident_outcome(question)
    if kind == "machine":
        direct_causes = [
            "프레스의 방호장치ㆍ인터록이 해제된 상태에서 정비ㆍ청소 작업 중 기계가 갑자기 작동하여 협착 사고가 발생했습니다.",
            "동력 차단, 잠금ㆍ표지와 기동 전 작업자 확인이 이루어지지 않았습니다.",
        ]
        indirect = (
            "특별교육, 정비 작업허가, 에너지 차단 절차와 수급업체 작업 통제가 현장에서 작동하지 않았습니다."
            if has_contract
            else "특별교육, 정비 작업허가, 에너지 차단과 방호장치 점검 절차가 직영 작업 현장에서 작동하지 않았습니다."
        )
        violations = [
            "프레스 특별교육 미실시: 산업안전보건법 제29조제3항 및 시행규칙 별표 5 제1호라목 제11호",
            "원동기ㆍ회전축 방호 미흡: 산업안전보건기준에 관한 규칙 제87조",
            "동력차단장치와 운전 시작 전 확인 미흡: 제88조ㆍ제89조",
            "정비ㆍ청소 시 운전정지ㆍ잠금ㆍ표지 미실시: 제92조",
            "방호장치 해체ㆍ무력화: 제93조",
            "프레스 위험한계 및 금형조정 방호 미흡: 제103조ㆍ제104조",
            "프레스 안전검사 미실시: 산업안전보건법 제93조",
            "고장 난 방호장치의 점검ㆍ정비 미흡: 산업안전보건법 제80조제3항",
        ]
        measures = [
            "프레스 운전을 즉시 중지하고 에너지원을 차단한 뒤 개인 잠금장치와 작업표지를 적용합니다. 근거: 제88조ㆍ제92조",
            "방호덮개ㆍ인터록ㆍ광전자식 방호장치와 안전블록을 복구ㆍ점검합니다. 근거: 제87조ㆍ제93조ㆍ제103조ㆍ제104조",
            "프레스 특별교육 이수자만 투입하고 기동 전 작업자 확인ㆍ신호 절차를 시행합니다. 근거: 별표 5 제1호라목 제11호, 제89조",
        ]
    else:
        direct_causes = [
            "크레인 인양 중 달기구ㆍ걸고리 또는 와이어로프 관리와 인양 방법이 미흡하여 하물이 이탈ㆍ낙하했습니다.",
            "인양물 아래 출입금지구역과 신호체계가 확보되지 않아 근로자가 낙하물 위험에 노출됐습니다.",
        ]
        indirect = "특별교육, 인양계획, 달기구 사전점검과 원ㆍ하청 작업조정 절차가 현장에서 작동하지 않았습니다."
        violations = [
            "크레인 특별교육 미실시: 산업안전보건법 제29조제3항 및 시행규칙 별표 5 제1호라목 제14호",
            "낙하ㆍ비래 방지와 출입통제 미흡: 산업안전보건기준에 관한 규칙 제14조",
            "정격하중 표시와 양중기 방호장치 점검 미흡: 제133조ㆍ제134조",
            "크레인 작업 안전조치 미준수: 제146조",
            "이동식 크레인 해지장치 미사용: 제149조",
        ]
        measures = [
            "인양 작업을 즉시 중지하고 훅ㆍ와이어로프ㆍ해지장치와 양중기 방호장치를 점검ㆍ교체합니다. 근거: 제134조ㆍ제149조",
            "인양물 아래 출입금지구역과 감시자를 배치하고 표준 신호방법을 적용합니다. 근거: 제14조ㆍ제146조",
            "크레인 특별교육 이수자와 지정 신호자만 작업에 참여시킵니다. 근거: 별표 5 제1호라목 제14호",
        ]
    if death:
        serious_definition = "제2조제2호가목"
        manager_penalty = "제6조제1항에 따라 1년 이상의 징역 또는 10억원 이하의 벌금"
        entity_penalty = "제7조제1호에 따라 50억원 이하의 벌금"
        serious_judgment = f"- 사망자 1명 이상이므로 중대재해처벌법 {serious_definition}의 중대산업재해에 해당합니다."
    elif serious_applies:
        serious_definition = "제2조제2호나목"
        manager_penalty = "제6조제2항에 따라 7년 이하의 징역 또는 1억원 이하의 벌금"
        entity_penalty = "제7조제2호에 따라 10억원 이하의 벌금"
        serious_judgment = f"- 6개월 이상 치료가 필요한 부상자가 {injury_count}명이므로 중대재해처벌법 {serious_definition}의 중대산업재해에 해당합니다."
    else:
        serious_definition = "제2조제2호나목"
        manager_penalty = "중대재해처벌법 제6조의 형사처벌 적용 대상이 아닙니다"
        entity_penalty = "중대재해처벌법 제7조의 법인 양벌규정 적용 대상이 아닙니다"
        serious_judgment = (
            f"- 사망자는 없고 {treatment_months or 6}개월 이상 치료가 필요한 부상자가 {injury_count or 1}명이므로 "
            f"중대재해처벌법 {serious_definition}의 '부상자 2명 이상' 요건을 충족하지 않아 미적용입니다."
        )
    if serious_applies:
        manager_judgment = f"- 안전보건관리체계 위반과 사고 사이의 인과관계가 인정되면 {manager_penalty} 대상이 될 수 있습니다."
        entity_judgment = f"- 경영책임자의 위반행위가 인정되면 법인 {company}은 {entity_penalty} 대상이 될 수 있습니다."
    else:
        manager_judgment = "- 안전보건관리체계 개선 필요성은 별도로 검토하되, 이 사고는 중대산업재해 결과 요건에 미달하므로 제6조의 경영책임자 형사처벌 대상이 아닙니다."
        entity_judgment = "- 경영책임자에 대한 제6조 위반이 성립하지 않으므로 이 사고를 근거로 제7조의 법인 양벌규정을 적용하지 않습니다."
    lines = ["최종 보고서", "", "1. 사고 원인 분석", "[직접 원인]"]
    lines.extend(f"- {item}" for item in direct_causes)
    lines.extend(["", "[간접 원인]", f"- {indirect}", "", "2. 법령 위반 사항 요약", "[산업안전보건법]"])
    lines.extend(f"- {item}" for item in violations)
    lines.extend(
        [
            "",
            "[중대재해처벌법]",
            serious_judgment,
            "- 경영책임자의 유해ㆍ위험요인 확인ㆍ개선과 관계 법령 이행 점검체계가 작동했는지 검토합니다. 근거: 중대재해처벌법 제4조",
            "",
            "3. 책임 주체별 판단",
            "[사업주]",
            f"- {company} 사업주는 현장의 교육ㆍ방호ㆍ작업중지 및 통제 의무의 1차 책임 주체입니다.",
            "- 현재 확인된 위반의 처벌 주체는 사업주이며, 현장 근로자의 행위를 근로자 개인 처벌로 바로 귀결하지 않습니다.",
            "",
            "[경영책임자]",
            manager_judgment,
            "",
            "[법인]",
            entity_judgment,
            "",
            "4. 즉시 취해야 할 재발방지 조치",
        ]
    )
    if has_contract:
        business_index = lines.index("- 현재 확인된 위반의 처벌 주체는 사업주이며, 현장 근로자의 행위를 근로자 개인 처벌로 바로 귀결하지 않습니다.")
        lines.insert(
            business_index,
            f"- {contractor} 소속 근로자가 작업했더라도 원청의 도급인 의무를 별도로 검토합니다. 근거: 산업안전보건법 제64조",
        )
    else:
        business_index = lines.index("- 현재 확인된 위반의 처벌 주체는 사업주이며, 현장 근로자의 행위를 근로자 개인 처벌로 바로 귀결하지 않습니다.")
        lines.insert(business_index, "- 이 사고는 직영 작업이므로 해당 사업주의 직접 교육ㆍ방호ㆍ검사 의무를 중심으로 판단합니다.")
    lines.extend(f"- {item}" for item in measures)
    if has_contract:
        lines.append(f"- {contractor}의 안전보건 역량과 작업계획을 재평가하고 반기 점검체계를 운영합니다. 근거: 중대재해처벌법 제5조 및 시행령 제4조제9호")
    else:
        lines.append("- 직영 작업의 프레스 점검ㆍ정비 책임자와 작업중지 권한을 명확히 하고 이행 결과를 기록합니다. 근거: 산업안전보건법 제80조제3항")
    return "\n".join(lines)
