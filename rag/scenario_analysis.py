"""LLM-assisted scenario fact extraction with deterministic validation."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from rag.falling_object_routing import is_masonry_falling_scenario
from rag.hot_work_routing import is_excavation_scenario, is_hot_work_scenario
from rag.schemas import AccidentScenario
from rag.v1_incident_routing import (
    extract_company_name,
    extract_contractor_name,
    is_machine_entanglement_scenario,
    is_struck_by_scenario,
)


PROFILE_SCHEMA_VERSION = 1


def scenario_text(scenario: AccidentScenario) -> str:
    return "\n".join(
        (
            f"[사고 개요]\n{scenario.overview.strip()}",
            f"[사고 경위]\n{scenario.details.strip()}",
            f"[근로자 현황]\n{scenario.workers.strip()}",
        )
    ).strip()


def scenario_hash(scenario: AccidentScenario) -> str:
    payload = json.dumps(scenario.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("모델 응답에서 JSON 객체를 찾지 못했습니다.")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("시나리오 분석 결과가 JSON 객체가 아닙니다.")
    return value


def _text_list(value: Any, *, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text[:180])
        if len(result) >= limit:
            break
    return result


def _uncertainty_list(value: Any) -> list[str]:
    legal_terms = ("법령", "조항", "처벌", "위반", "법적", "확보의무")
    return [item for item in _text_list(value, limit=5) if not any(term in item for term in legal_terms)]


def _validated_uncertainties(raw: Any, text: str, contract_structure: str) -> list[str]:
    items = _uncertainty_list(raw)
    if contract_structure != "확인 필요":
        items = [item for item in items if "계약 구조" not in item]
    compact = re.sub(r"\s+", "", text)
    if "경영책임자:" in compact and "대표이사" in compact and "관여" in compact:
        items = [item for item in items if not ("경영책임자" in item and "역할" in item)]
    return items


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, min(int(value), 10000))
    except (TypeError, ValueError):
        return 0


def _explicit_count(text: str, word: str) -> int | None:
    if word == "사망":
        patterns = (r"사망자\s*[:：]?\s*(\d+)\s*명", r"(\d+)\s*명\s*(?:이\s*)?사망")
    elif word == "부상":
        patterns = (r"부상자\s*[:：]?\s*(\d+)\s*명", r"(\d+)\s*명\s*(?:이\s*)?부상")
    else:
        patterns = (
            r"6개월\s*이상[^.\n]{0,30}?부상자?\s*[:：]?\s*(\d+)\s*명",
            r"(\d+)\s*명[^.\n]{0,30}?6개월\s*이상",
        )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _contract_structure(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    has_contract = "도급" in compact
    has_dispatch = "파견" in compact
    if has_contract and has_dispatch:
        return "도급/파견 혼재"
    if has_dispatch:
        return "파견"
    if has_contract:
        return "도급"
    if "직영" in compact:
        return "직영"
    return "확인 필요"


def _ensure_issue(items: list[str], keyword: str, label: str) -> None:
    if not any(keyword in item for item in items):
        items.append(label)


def _validated_missing_controls(raw: Any, text: str) -> list[str]:
    controls = _text_list(raw)
    compact = re.sub(r"\s+", "", text)
    if "발열검사" in compact and any(term in compact for term in ("미운영", "이루어지지않", "미실시")):
        _ensure_issue(controls, "발열검사", "발열검사ㆍ분리보관 절차 미운영")
    if "위험성평가" in compact and any(term in compact for term in ("형식적", "미반영")):
        _ensure_issue(controls, "위험성평가", "실제 위험요인을 반영한 위험성평가 미실시")
    if any(term in compact for term in ("유사한전지발열", "유사발열", "반복사고")) and any(
        term in compact for term in ("계속진행", "작업지속", "전조무시")
    ):
        _ensure_issue(controls, "작업중지", "반복 사고 전조 확인 후 작업중지 미실시")
    return controls[:12]


def _deterministic_route(text: str) -> tuple[str, list[str]]:
    if is_masonry_falling_scenario(text):
        return "물체맞음", ["masonry_falling"]
    compact = re.sub(r"\s+", "", text)
    if any(term in compact for term in ("화재", "폭발", "발열")) and any(
        term in compact for term in ("리튬", "전지", "발화성", "유독가스")
    ):
        return "화재ㆍ폭발", ["industrial_fire_explosion"]
    if is_hot_work_scenario(text):
        return "화재ㆍ폭발", ["fire_welding"]
    if is_excavation_scenario(text):
        return "붕괴ㆍ매몰", ["collapse_excavation"]
    if is_machine_entanglement_scenario(text):
        return "끼임", ["machine_entanglement"]
    if is_struck_by_scenario(text):
        return "맞음ㆍ낙하", ["struck_by"]
    return "기타", []


def validate_scenario_profile(raw: dict[str, Any], scenario: AccidentScenario) -> dict[str, Any]:
    text = scenario_text(scenario)
    accident_type, routing_hints = _deterministic_route(text)
    death_count = _explicit_count(text, "사망")
    injury_count = _explicit_count(text, "부상")
    long_term_count = _explicit_count(text, "6개월 이상")
    company = extract_company_name(text, "확인 필요")
    contractor = extract_contractor_name(text, "확인 필요")
    contract_structure = _contract_structure(text)

    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "accident_type": accident_type if routing_hints else str(raw.get("accident_type") or "기타").strip()[:40],
        "work_types": _text_list(raw.get("work_types")),
        "company": company,
        "contractor": contractor,
        "contract_structure": contract_structure,
        "death_count": death_count if death_count is not None else _nonnegative_int(raw.get("death_count")),
        "injury_count": injury_count if injury_count is not None else _nonnegative_int(raw.get("injury_count")),
        "long_term_injury_count": long_term_count,
        "hazards": _text_list(raw.get("hazards")),
        "missing_controls": _validated_missing_controls(raw.get("missing_controls"), text),
        "uncertainties": _validated_uncertainties(raw.get("uncertainties"), text, contract_structure),
        "routing_hints": routing_hints,
        "validation": {
            "method": "llm_extraction_plus_deterministic_rules",
            "legal_articles_decided_by_llm": False,
        },
    }


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return _nonnegative_int(value)


def _edited_routing_hints(profile: dict[str, Any], current: dict[str, Any]) -> list[str]:
    accident_type = str(profile.get("accident_type") or "").strip()
    context = " ".join(
        [
            accident_type,
            *(_text_list(profile.get("work_types"))),
            *(_text_list(profile.get("hazards"))),
        ]
    )
    compact = re.sub(r"\s+", "", context)
    current_hints = _text_list(current.get("routing_hints"), limit=4)

    if any(term in compact for term in ("화재", "폭발", "발열")):
        if any(term in compact for term in ("용접", "화기", "아세틸렌")):
            return ["fire_welding"]
        if "fire_welding" in current_hints and not any(term in compact for term in ("리튬", "전지", "유독가스")):
            return ["fire_welding"]
        return ["industrial_fire_explosion"]
    if any(term in compact for term in ("붕괴", "매몰", "굴착")):
        return ["collapse_excavation"]
    if any(term in compact for term in ("끼임", "협착", "프레스")):
        return ["machine_entanglement"]
    if any(term in compact for term in ("물체맞음", "물체에맞음", "낙하물", "조적")):
        if "masonry_falling" in current_hints or any(term in compact for term in ("조적", "벽돌", "발끝막이")):
            return ["masonry_falling"]
        return ["struck_by"]
    return current_hints


def normalize_user_scenario_profile(raw: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Validate user-corrected facts without accepting legal conclusions."""
    if not isinstance(raw, dict):
        raise ValueError("수정할 분석 정보가 올바르지 않습니다.")
    if not isinstance(current, dict):
        current = {}

    allowed_contracts = {"직영", "도급", "파견", "도급/파견 혼재", "확인 필요"}
    contract_structure = str(raw.get("contract_structure") or "확인 필요").strip()
    if contract_structure not in allowed_contracts:
        contract_structure = "확인 필요"

    profile = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "accident_type": str(raw.get("accident_type") or "기타").strip()[:40] or "기타",
        "work_types": _text_list(raw.get("work_types")),
        "company": str(raw.get("company") or "확인 필요").strip()[:120] or "확인 필요",
        "contractor": str(raw.get("contractor") or "확인 필요").strip()[:120] or "확인 필요",
        "contract_structure": contract_structure,
        "death_count": _nonnegative_int(raw.get("death_count")),
        "injury_count": _nonnegative_int(raw.get("injury_count")),
        "long_term_injury_count": _optional_nonnegative_int(raw.get("long_term_injury_count")),
        "hazards": _text_list(raw.get("hazards")),
        "missing_controls": _text_list(raw.get("missing_controls")),
        "uncertainties": _uncertainty_list(raw.get("uncertainties")),
    }
    profile["routing_hints"] = _edited_routing_hints(profile, current)
    profile["validation"] = {
        "method": "user_corrected_after_llm",
        "legal_articles_decided_by_llm": False,
        "user_edited": True,
    }
    return profile


def analyze_scenario(scenario: AccidentScenario) -> tuple[dict[str, Any], str]:
    """Call the configured model once, then validate its fact-only JSON output."""
    from rag.chatbot import call_llm_blocking

    prompt = f"""/no_think
다음 건설현장 사고 시나리오에서 사실관계만 구조화하라.
법령 조항, 처벌 수위, 위반 여부를 판단하거나 생성하지 마라.
원문에 없는 회사명, 인원, 작업, 장비를 추정하지 마라.
확실하지 않은 내용은 uncertainties에 넣어라.

반드시 아래 키만 가진 JSON 객체 하나만 출력하라.
{{
  "accident_type": "사고 유형",
  "work_types": ["작업 종류, 최대 3개"],
  "company": "원청 또는 사업주명",
  "contractor": "수급ㆍ협력업체명",
  "contract_structure": "직영/도급/확인 필요",
  "death_count": 0,
  "injury_count": 0,
  "long_term_injury_count": 0,
  "hazards": ["유해ㆍ위험요인, 최대 5개"],
  "missing_controls": ["시나리오에 명시된 미조치 사항, 최대 5개"],
  "uncertainties": ["추가 확인이 필요한 사실, 최대 3개"]
}}

{scenario_text(scenario)}"""
    raw_response = call_llm_blocking(
        [
            {"role": "system", "content": "당신은 사고 사실 추출기다. 설명 없이 유효한 JSON만 출력한다."},
            {"role": "user", "content": prompt},
        ],
        {"temperature": 0.0, "num_predict": 450},
    )
    return validate_scenario_profile(_json_object(raw_response), scenario), raw_response


def format_scenario_profile(profile: dict[str, Any] | None) -> str:
    if not profile:
        return ""
    return json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
