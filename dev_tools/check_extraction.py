"""
PDF 표 추출 디버깅 도구.

사용법:
  python dev_tools/check_extraction.py --page 82          # 특정 페이지 raw 표 확인
  python dev_tools/check_extraction.py --page 122 --df    # DataFrame 변환 결과 확인
  python dev_tools/check_extraction.py --keyword 벤젠     # 키워드 포함 행 검색
  python dev_tools/check_extraction.py --page 82 --chunk  # 청킹 결과까지 확인
"""
import argparse
import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import pdfplumber
from pathlib import Path
from rag.table_extraction import raw_table_to_df, PDFPLUMBER_TABLE_SETTINGS
from rag.table_chunking import chunk_by_item, chunk_by_row

PDF_PATH = Path("data/laws/02_산업안전보건법/산업안전보건법_시행규칙.pdf")


def check_page(page_num: int, show_df: bool = False, show_chunks: bool = False):
    with pdfplumber.open(str(PDF_PATH)) as pdf:
        page = pdf.pages[page_num - 1]
        tables = page.extract_tables(PDFPLUMBER_TABLE_SETTINGS)
        print(f"\n=== p.{page_num}: {len(tables)}개 표 ===")
        for t_idx, raw in enumerate(tables):
            print(f"\n--- 표 {t_idx} raw ({len(raw)}행 x {len(raw[0]) if raw else 0}열) ---")
            for r_idx, row in enumerate(raw[:6]):
                cells = [repr(str(c)[:40]) if c else '""' for c in row]
                print(f"  행{r_idx:02d}: {cells}")
            if len(raw) > 6:
                print(f"  ... {len(raw)-6}행 더")

            if show_df or show_chunks:
                df = raw_table_to_df(raw)
                if df.empty:
                    continue
                print(f"\n  DataFrame: {len(df)}행, 컬럼={list(df.columns)}")
                for i, row in df.head(4).iterrows():
                    print(f"  행{i}: { {k: str(v)[:40] for k, v in row.items()} }")

            if show_chunks:
                df = raw_table_to_df(raw)
                chunks = chunk_by_item(df, {"page": page_num, "table_index": t_idx})
                print(f"\n  청킹 결과: {len(chunks)}개")
                for c in chunks[:4]:
                    print(f"  [{c.metadata.get('chunk_strategy')}] item={str(c.metadata.get('item_number',''))[:30]}")
                    print(f"    {c.text[:100]}")


def search_keyword(keyword: str):
    with pdfplumber.open(str(PDF_PATH)) as pdf:
        print(f"\n=== '{keyword}' 검색 ===")
        for page_num, page in enumerate(pdf.pages, 1):
            tables = page.extract_tables(PDFPLUMBER_TABLE_SETTINGS) or []
            for t_idx, raw in enumerate(tables):
                df = raw_table_to_df(raw)
                if df.empty:
                    continue
                for i, row in df.iterrows():
                    if any(keyword in str(v) for v in row.values):
                        print(f"\np.{page_num} 표{t_idx} 행{i}:")
                        print(f"  컬럼: {list(df.columns)}")
                        print(f"  값:   { {k: str(v)[:60] for k, v in row.items()} }")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", type=int, help="확인할 페이지 번호")
    parser.add_argument("--df", action="store_true", help="DataFrame 변환 결과 표시")
    parser.add_argument("--chunk", action="store_true", help="청킹 결과까지 표시")
    parser.add_argument("--keyword", type=str, help="검색할 키워드")
    args = parser.parse_args()

    if args.keyword:
        search_keyword(args.keyword)
    elif args.page:
        check_page(args.page, show_df=args.df, show_chunks=args.chunk)
    else:
        parser.print_help()
