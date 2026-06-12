"""
주요 쿼리 동작 검증 스크립트.

사용법:
  python scripts/test_queries.py                 # 기본 4개 쿼리
  python scripts/test_queries.py --query "카드뮴 TWA"   # 단일 쿼리
"""
import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import argparse
from rag.chatbot import rag_chat
from rag.schemas import ChatRequest

DEFAULT_QUERIES = [
    "벤젠(Benzene)의 시간가중평균값(TWA) ppm 기준은?",
    "석면(Asbestos)의 허용기준 값과 단위는?",
    "굴착면의 높이가 2미터 이상인 지반 굴착작업 시 특별교육 내용을 모두 알려줘",
    "건설현장에서 높이 3미터 굴착작업을 진행 중인데 특별안전교육을 실시하지 않았다. 위반인가?",
]

SEP = "=" * 60


def run_query(question: str):
    print(f"\n{SEP}\n질문: {question}\n{SEP}", flush=True)
    resp = rag_chat(ChatRequest(question=question))
    print(f"[답변]\n{resp.answer}", flush=True)
    print(f"\n[소스 상위 3개]", flush=True)
    for i, s in enumerate(resp.sources[:3], 1):
        m = s.metadata
        item = str(m.get("item_number", ""))
        print(f"  [{i}] p.{m.get('page','')} {m.get('chunk_strategy','')} "
              f"item={item[:30] if item else ''}", flush=True)
        print(f"       {s.content[:80]}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="단일 쿼리 실행")
    args = parser.parse_args()

    queries = [args.query] if args.query else DEFAULT_QUERIES
    for q in queries:
        run_query(q)
    print(f"\n{SEP}\n완료", flush=True)


if __name__ == "__main__":
    main()
