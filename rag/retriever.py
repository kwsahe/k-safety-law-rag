"""
retriever.py
ChromaDB에서 질문과 유사한 법령 청크를 검색하는 모듈.
"""

import logging
from typing import List, Optional

from rag.config import RAG_TOP_K
from rag.schemas import SourceDoc
from rag.embedder import get_embedder
from rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: Optional[int] = None) -> List[SourceDoc]:
    """
    질문을 임베딩한 뒤 ChromaDB에서 유사 법령 청크를 검색한다.

    Args:
        query: 사용자 질문 문자열
        top_k: 반환할 최대 문서 수. None이면 config.RAG_TOP_K 사용.

    Returns:
        유사도 높은 순으로 정렬된 SourceDoc 리스트
    """
    k = top_k if top_k is not None else RAG_TOP_K

    embedder = get_embedder()
    vector_store = get_vector_store()

    query_embedding = embedder.encode(query)
    results = vector_store.query(query_embedding=query_embedding, n_results=k)

    return _parse_results(results)


def _parse_results(results: dict) -> List[SourceDoc]:
    """ChromaDB 쿼리 결과 딕셔너리를 SourceDoc 리스트로 변환한다."""
    docs = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for content, metadata, distance in zip(documents, metadatas, distances):
        enriched_metadata = {**metadata, "score": _distance_to_score(distance)}
        docs.append(SourceDoc(content=content, metadata=enriched_metadata))

    return docs


def _distance_to_score(distance: float) -> float:
    """cosine distance → 유사도 점수 변환. ChromaDB cosine 거리는 0(동일)~2(반대)."""
    return round(1.0 - distance, 4)
