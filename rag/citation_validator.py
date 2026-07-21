"""Citation and source consistency checks for RAG answers.

The validator is intentionally rule-based.  It does not try to grade legal
reasoning; it checks whether important citations in the answer are supported
by the returned source documents and flags recurring source-mixup patterns.
"""

from __future__ import annotations

import re
from typing import Any


LAW_NAMES = (
    "중대재해처벌법 시행령",
    "중대재해처벌법",
    "산업안전보건법 시행규칙",
    "산업안전보건법 시행령",
    "산업안전보건기준에 관한 규칙",
    "산업안전보건법",
)

ARTICLE_RE = re.compile(r"제\s*\d+\s*조(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?(?:\s*[가-힣]\s*목)?")
ANNEX_RE = re.compile(r"별표\s*\d+(?:\s*제\s*\d+\s*호)?")


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", _text(value).replace("_", " "))


def _source_blob(source: Any) -> str:
    metadata = getattr(source, "metadata", {}) or {}
    parts = [
        metadata.get("law_name", ""),
        metadata.get("article", ""),
        metadata.get("annex", ""),
        metadata.get("item_number", ""),
        metadata.get("citation_page", ""),
        metadata.get("page", ""),
        getattr(source, "content", ""),
    ]
    return _norm(" ".join(_text(part) for part in parts))


def _label(law_name: str, ref: str) -> str:
    law = law_name.strip()
    ref = re.sub(r"\s+", " ", ref).strip()
    return f"{law} {ref}".strip()


def _extract_citations(answer: str) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    law_pattern = re.compile("|".join(re.escape(name) for name in LAW_NAMES))
    for line in answer.splitlines():
        matches = list(law_pattern.finditer(line))
        for index, match in enumerate(matches):
            law_name = match.group(0)
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
            window = line[match.start() : end]
            refs = ARTICLE_RE.findall(window) + ANNEX_RE.findall(window)
            for ref in refs:
                key = (_norm(law_name), _norm(ref))
                if key in seen:
                    continue
                seen.add(key)
                citations.append({"law_name": law_name, "ref": ref, "label": _label(law_name, ref)})
    return citations


def _source_supports(source: Any, law_name: str, ref: str) -> bool:
    blob = _source_blob(source)
    law = _norm(law_name)
    ref_norm = _norm(ref)
    if ref_norm not in blob:
        return False

    # Some synthetic/direct sources only store a compact law label.  Requiring
    # the full law name would create false warnings, so compare by family.
    if "중대재해처벌법" in law:
        return "중대재해처벌법" in blob
    if "산업안전보건기준에관한규칙" in law:
        return "산업안전보건기준에관한규칙" in blob
    if "산업안전보건법" in law:
        return "산업안전보건법" in blob or "산업안전보건기준에관한규칙" in blob
    return True


def _find_support(sources: list[Any], law_name: str, ref: str) -> int | None:
    for index, source in enumerate(sources, start=1):
        if _source_supports(source, law_name, ref):
            return index
    return None


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _required_rules(answer: str) -> list[dict[str, str]]:
    compact = _norm(answer)
    rules: list[dict[str, str]] = []
    if "법인" in compact and re.search(r"50\s*억|50억원", answer):
        rules.append(
            {
                "id": "corporate_50b_fine",
                "label": "법인 50억원 이하 벌금",
                "law_name": "중대재해처벌법",
                "ref": "제7조",
            }
        )
    if "경영책임자" in compact and "1년이상" in compact:
        rules.append(
            {
                "id": "manager_criminal_penalty",
                "label": "경영책임자 형사처벌",
                "law_name": "중대재해처벌법",
                "ref": "제6조제1항",
            }
        )
    if "경영책임자" in compact and "7년이하" in compact:
        rules.append(
            {
                "id": "manager_injury_criminal_penalty",
                "label": "경영책임자 부상 중대산업재해 형사처벌",
                "law_name": "중대재해처벌법",
                "ref": "제6조제2항",
            }
        )
    if ("손해액" in compact or "손해배상" in compact) and "5배" in compact:
        rules.append(
            {
                "id": "punitive_damages",
                "label": "손해액 5배 이내 배상책임",
                "law_name": "중대재해처벌법",
                "ref": "제15조",
            }
        )
    scaffold_training_claim = "별표5제23호" in compact
    if scaffold_training_claim and ("특별안전교육" in compact or "특별교육" in compact):
        rules.append(
            {
                "id": "scaffold_special_education",
                "label": "비계 작업 특별안전교육",
                "law_name": "산업안전보건법 시행규칙",
                "ref": "별표 5 제23호",
            }
        )
    contract_liability_claim = _contains_any(compact, ("도급", "원청", "협력업체", "수급인")) and _contains_any(
        compact,
        ("책임이성립", "책임주체", "안전보건조치", "산업재해예방조치", "도급인의무", "원청의의무"),
    )
    if contract_liability_claim:
        if "산업안전보건법" in compact:
            rules.append(
                {
                    "id": "prime_contractor_osh_act",
                    "label": "도급인의 안전보건조치 의무",
                    "law_name": "산업안전보건법",
                    "ref": "제64조",
                }
            )
        if "중대재해처벌법" in compact:
            rules.append(
                {
                    "id": "prime_contractor_serious_act",
                    "label": "도급 등 관계의 안전보건 확보의무",
                    "law_name": "중대재해처벌법",
                    "ref": "제5조",
                }
            )

    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for rule in rules:
        key = (_norm(rule["law_name"]), _norm(rule["ref"]))
        if key not in seen:
            seen.add(key)
            unique.append(rule)
    return unique


def _detect_known_mixups(answer: str) -> list[str]:
    warnings: list[str] = []
    for line in answer.splitlines():
        compact = _norm(line)
        if "법인" in compact and ("50억" in compact or "50억원" in compact) and "별표4" in compact:
            warnings.append("법인 50억원 이하 벌금의 근거가 별표 4로 표기되었습니다. 정합 근거는 중대재해처벌법 제7조입니다.")
            break
    compact_answer = _norm(answer)
    if "법인" in compact_answer and ("50억" in compact_answer or "50억원" in compact_answer):
        if "별표4" in compact_answer and "제7조" not in compact_answer:
            warnings.append("법인 벌금 근거에서 중대재해처벌법 제7조가 누락되고 별표 4가 사용되었습니다.")
    return warnings


def validate_answer_citations(answer: str, sources: list[Any]) -> dict[str, Any]:
    citations = _extract_citations(answer)
    checked_citations: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for citation in citations:
        matched = _find_support(sources, citation["law_name"], citation["ref"])
        item = {
            **citation,
            "supported": matched is not None,
            "source_index": matched,
        }
        checked_citations.append(item)
        if matched is None:
            unsupported.append(item)

    required: list[dict[str, Any]] = []
    missing_required: list[dict[str, Any]] = []
    for rule in _required_rules(answer):
        matched = _find_support(sources, rule["law_name"], rule["ref"])
        item = {
            **rule,
            "supported": matched is not None,
            "source_index": matched,
        }
        required.append(item)
        if matched is None:
            missing_required.append(item)

    warnings = _detect_known_mixups(answer)
    if warnings or missing_required:
        status = "fail"
    elif unsupported:
        status = "warn"
    else:
        status = "pass"

    if status == "pass":
        summary = "답변의 핵심 조항이 참고 근거와 일치합니다."
    elif status == "warn":
        summary = "답변에 참고 근거에서 직접 확인되지 않은 출처 표기가 있습니다."
    else:
        summary = "답변의 핵심 출처 또는 반복 오류 패턴을 확인해야 합니다."

    return {
        "status": status,
        "summary": summary,
        "citations": checked_citations,
        "required": required,
        "unsupported": unsupported,
        "missing_required": missing_required,
        "warnings": warnings,
    }
