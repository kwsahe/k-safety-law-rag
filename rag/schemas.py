"""
schemas.py
RAG 모듈의 데이터 통신 규격을 정의하는 Pydantic 모델.
"""

from pydantic import BaseModel
from typing import List, Optional

class AccidentScenario(BaseModel):
    """사고 시나리오 배경 정보"""
    overview: str = ""   # 사고 개요
    details: str = ""    # 사고 경위
    workers: str = ""    # 근로자 현황


class ChatRequest(BaseModel):
    """사용자의 질문을 받는 데이터 규격"""
    question: str
    scenario: Optional[AccidentScenario] = None
    use_direct_answers: bool = True

class SourceDoc(BaseModel):
    """검색된 법령 문서를 담는 규격"""
    content: str
    metadata: dict  # 법령 이름, 조항 번호 등

class ChatResponse(BaseModel):
    """챗봇의 최종 답변 데이터 규격"""
    answer: str
    sources: List[SourceDoc]
    
class LawMapperRequest(BaseModel):
    """사고 이벤트 유형으로 법령을 찾을 때 사용하는 규격"""
    event_type: str
