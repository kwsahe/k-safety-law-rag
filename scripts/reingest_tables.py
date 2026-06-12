"""테이블 벡터 DB 재적재 스크립트."""
import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import argparse
from rag.table_retriever import ingest_tables
from rag.table_vector_store import get_table_vector_store


def main():
    parser = argparse.ArgumentParser(description="테이블 벡터 DB 재적재")
    parser.add_argument("--strategy", default="item", choices=["row", "item", "table", "metadata"])
    parser.add_argument("--reset", action="store_true", default=True)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    print(f"재적재 시작 (strategy={args.strategy}, reset={args.reset})", flush=True)
    ingest_tables(strategy=args.strategy, reset=args.reset, max_files=args.max_files)
    count = get_table_vector_store().collection.count()
    print(f"완료: {count}개 청크", flush=True)


if __name__ == "__main__":
    main()
