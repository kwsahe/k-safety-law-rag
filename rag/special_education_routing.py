"""Shared routing rules for special-safety-education work types."""

from __future__ import annotations

import re


SPECIAL_EDUCATION_TERMS = (
    "특별교육",
    "특별안전교육",
    "교육내용",
    "교육사항",
    "미실시",
    "미이수",
)

WELDING_TERMS = (
    "용접",
    "가스용접",
    "가스집합용접",
    "아세틸렌",
    "산소-아세틸렌",
    "산소아세틸렌",
    "용단",
)

FRAME_OBJECT_TERMS = (
    "철골",
    "골조",
    "금속제부재",
    "건축물의골조",
)

FRAME_ACTION_TERMS = (
    "조립",
    "해체",
    "변경",
    "건립",
)


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def has_special_education_signal(value: str) -> bool:
    compact = compact_text(value)
    return any(term in compact for term in SPECIAL_EDUCATION_TERMS)


def has_welding_work_signal(value: str) -> bool:
    compact = compact_text(value)
    return any(term in compact for term in WELDING_TERMS)


def has_frame_assembly_signal(value: str) -> bool:
    """Require both a frame object and an assembly-type action.

    Welding and repair of an existing steel structure are not frame assembly
    work merely because the scenario contains the noun "철골".
    """
    compact = compact_text(value)
    if has_welding_work_signal(compact):
        return False
    has_object = any(term in compact for term in FRAME_OBJECT_TERMS)
    has_action = any(term in compact for term in FRAME_ACTION_TERMS)
    return has_object and has_action


def is_welding_special_education_query(value: str) -> bool:
    return has_welding_work_signal(value) and has_special_education_signal(value)
