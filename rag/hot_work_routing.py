"""Shared intent rules for welding, fire-risk, and ventilation controls."""

from __future__ import annotations

import re


HOT_WORK_ISSUE_GROUPS = (
    ("화기작업허가", "작업허가", "허가미발급", "허가서"),
    ("환기조치", "환기미실시", "환풍기", "배풍기"),
    ("보호구미착용", "보호면", "방독마스크", "보호구"),
    ("동시작업", "동시작업통제", "작업지휘자", "용접과도장", "용접ㆍ도장"),
)


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def hot_work_issue_signals(value: str) -> list[str]:
    compact = compact_text(value)
    signals: list[str] = []
    for group in HOT_WORK_ISSUE_GROUPS:
        match = next((term for term in group if term in compact), "")
        if match:
            signals.append(match)
    return signals


def is_hot_work_controls_question(value: str) -> bool:
    compact = compact_text(value)
    signals = hot_work_issue_signals(compact)
    asks_judgment = any(term in compact for term in ("위반", "조항", "판단", "나열", "모두"))
    return len(signals) >= 2 and asks_judgment


def is_hot_work_scenario(value: str) -> bool:
    """Identify a welding/solvent fire scenario without relying on company names."""
    compact = compact_text(value)
    has_ignition_work = any(term in compact for term in ("용접", "화기작업", "아세틸렌", "용단"))
    has_fire_risk = any(term in compact for term in ("화재", "폭발", "인화성", "유기용제", "용제증기", "도장"))
    return has_ignition_work and has_fire_risk


def is_excavation_scenario(value: str) -> bool:
    compact = compact_text(value)
    has_excavation = any(term in compact for term in ("굴착", "토공", "굴착면", "흙막이", "지보공"))
    has_collapse_risk = any(term in compact for term in ("붕괴", "매몰", "토사", "기울기", "변형"))
    return has_excavation and has_collapse_risk


def extract_company_name(value: str, default: str = "해당 사업장") -> str:
    match = re.search(r"\(주\)\s*([가-힣A-Za-z0-9_-]+)", value or "")
    if match:
        name = match.group(1)
        if name.endswith("의"):
            name = name[:-1]
        return f"(주){name}"
    return default


def extract_contractor_name(value: str, default: str = "수급업체") -> str:
    patterns = (
        r"(?:외부\s*업체|협력\s*업체|수급\s*업체|하청\s*업체)\s*([가-힣A-Za-z0-9_-]+사)",
        r"([A-Za-z가-힣0-9_-]+사)에게?\s*도급",
    )
    for pattern in patterns:
        match = re.search(pattern, value or "")
        if match:
            return match.group(1)
    return default
