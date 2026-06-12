"""
law_mapper.py
사고 이벤트 유형(event_type) → 검색 키워드 매핑 모듈.
이벤트 유형별로 미리 정의된 키워드를 ChromaDB 검색에 활용한다.
"""

import logging
from typing import List

from rag.retriever import retrieve
from rag.schemas import SourceDoc

logger = logging.getLogger(__name__)

# 이벤트 유형 → 검색 키워드 매핑 테이블
EVENT_TO_KEYWORDS: dict[str, str] = {
    "helmet_off":    "안전모 미착용 보호구 착용 의무",
    "vest_off":      "안전조끼 고시인 의복 보호구 착용",
    "fall":          "추락 방지 안전난간 작업발판 안전망",
    "intrusion":     "위험 장소 출입 금지 접근 제한",
    "wind_warning":  "강풍 시 작업 중지 타워크레인 풍속 기준",
    "earthquake":    "지진 발생 시 양중기 점검 작업 중지",
    "other":         "산업재해 안전조치 사업주 의무",
}

# 이벤트 유형 → 사람이 읽기 좋은 한국어 설명
EVENT_LABELS: dict[str, str] = {
    "helmet_off":   "안전모 미착용",
    "vest_off":     "안전조끼 미착용",
    "fall":         "추락 위험",
    "intrusion":    "위험 구역 침입",
    "wind_warning": "강풍 경보",
    "earthquake":   "지진 발생",
    "other":        "기타 사고",
}


def get_laws_for_event(event_type: str, top_k: int = 3) -> List[SourceDoc]:
    """
    이벤트 유형에 해당하는 관련 법령 청크를 검색한다.

    Args:
        event_type: 사고 이벤트 유형 문자열 (예: "helmet_off")
        top_k: 반환할 최대 문서 수

    Returns:
        유사도 높은 순으로 정렬된 SourceDoc 리스트.
        알 수 없는 event_type이면 "other" 키워드로 폴백.
    """
    keyword = EVENT_TO_KEYWORDS.get(event_type)
    if keyword is None:
        logger.warning("알 수 없는 event_type '%s' — 'other' 키워드로 폴백합니다.", event_type)
        keyword = EVENT_TO_KEYWORDS["other"]

    logger.debug("event_type='%s' → 키워드='%s'", event_type, keyword)
    return retrieve(query=keyword, top_k=top_k)


def get_event_summary(event_type: str, camera_id: str = "", timestamp: str = "") -> str:
    """
    이벤트 정보를 한 줄 요약 문자열로 반환한다 (보고서·로그용).

    Args:
        event_type: 사고 이벤트 유형 문자열
        camera_id: 카메라 식별자 (선택)
        timestamp: 발생 시각 문자열 (선택)

    Returns:
        예: "[강풍 경보] 카메라 CAM-01 / 2026-05-25 14:30:00"
    """
    label = EVENT_LABELS.get(event_type, event_type)
    parts = [f"[{label}]"]
    if camera_id:
        parts.append(f"카메라 {camera_id}")
    if timestamp:
        parts.append(str(timestamp))
    return " / ".join(parts)
