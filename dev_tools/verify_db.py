"""
ChromaDB 적재 상태 및 청킹 라우팅 검증 도구.

사용법:
  python dev_tools/verify_db.py                    # DB 청크 수 + 샘플 확인
  python dev_tools/verify_db.py --query "벤젠 TWA" # 쿼리 검색 결과 확인
  python dev_tools/verify_db.py --chunk-routing    # 청킹 전략 라우팅 확인
"""
import argparse
import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())


def check_db(query: str | None = None):
    from rag.table_vector_store import get_table_vector_store
    from rag.embedder import Embedder

    vs = get_table_vector_store()
    count = vs.collection.count()
    print(f"총 청크 수: {count}")

    if query:
        emb = Embedder()
        q_vec = emb.encode(query)
        results = vs.collection.query(
            query_embeddings=[q_vec], n_results=5,
            include=["documents", "metadatas"]
        )
        print(f"\n쿼리: {query}")
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
            print(f"\n[{i}] p.{meta.get('page','')} strategy={meta.get('chunk_strategy','')} "
                  f"item={str(meta.get('item_number',''))[:35]}")
            print(f"    {doc[:120]}")
    else:
        # 전략별 청크 수 집계
        all_meta = vs.collection.get(include=["metadatas"])["metadatas"]
        from collections import Counter
        strategies = Counter(m.get("chunk_strategy", "unknown") for m in all_meta)
        print(f"전략별 청크 수: {dict(strategies)}")


def check_chunk_routing():
    import pandas as pd
    from rag.table_chunking import chunk_by_item

    # 노출기준 표 (TWA_ppm 컬럼) → row 전략이어야 함
    df_exp = pd.DataFrame([
        ["12. 벤젠(Benzene)", "", "0.5", "", "2.5", ""],
    ], columns=["유해인자", "세부구분", "TWA_ppm", "TWA_mg_m3", "STEL_ppm", "STEL_mg_m3"])
    chunks = chunk_by_item(df_exp, {"page": 122})
    result = chunks[0].metadata["chunk_strategy"] if chunks else "없음"
    status = "✅" if result == "row" else "❌"
    print(f"{status} 노출기준 표 → {result} (기대: row)")

    # 특별교육 표 (col_0, col_1) → item 전략이어야 함
    df_edu = pd.DataFrame([
        ["19. 굴착면의 높이가 2미터 이상...", "○ 지반의 형태 ○ 붕괴재해"],
        ["20. 흙막이 지보공...", "○ 작업안전 점검"],
    ], columns=["col_0", "col_1"])
    chunks2 = chunk_by_item(df_edu, {"page": 82})
    result2 = chunks2[0].metadata["chunk_strategy"] if chunks2 else "없음"
    status2 = "✅" if result2 == "item" else "❌"
    print(f"{status2} 특별교육 표 → {result2} (기대: item)")

    # 일반 표 (번호 없음) → row 전략이어야 함
    df_plain = pd.DataFrame([["가", "나", "다"], ["1", "2", "3"]], columns=["A", "B", "C"])
    chunks3 = chunk_by_item(df_plain, {})
    result3 = chunks3[0].metadata["chunk_strategy"] if chunks3 else "없음"
    status3 = "✅" if result3 == "row" else "❌"
    print(f"{status3} 일반 표 → {result3} (기대: row)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="검색 쿼리")
    parser.add_argument("--chunk-routing", action="store_true", help="청킹 라우팅 확인")
    args = parser.parse_args()

    if args.chunk_routing:
        check_chunk_routing()
    else:
        check_db(query=args.query)
