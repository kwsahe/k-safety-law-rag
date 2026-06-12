"""Convenience runner for the integrated law-table RAG workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.table_report import generate_report
from rag.table_retriever import ingest_tables, search_table_chunks

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def run_pipeline(
    query: str | None = None,
    skip_ingest: bool = False,
    skip_report: bool = False,
    reset: bool = False,
    strategy: str = "item",
    max_files: int | None = None,
    max_tables: int | None = None,
    html_only: bool = False,
) -> tuple[list[dict] | None, Path | None, Path | None]:
    search_results: list[dict] | None = None
    html_path: Path | None = None
    pdf_path: Path | None = None

    if not skip_ingest:
        ingest_tables(
            strategy=strategy,
            reset=reset,
            max_files=max_files,
            max_tables=max_tables,
        )

    if query:
        search_results = search_table_chunks(query)
        print(f"\nquery: {query}\n" + "=" * 80)
        for index, hit in enumerate(search_results, start=1):
            metadata = hit["metadata"]
            print(
                f"\n[{index}] score={hit['score']} "
                f"{metadata.get('law_name', '')} p.{metadata.get('page', '')}"
            )
            print(str(hit["text"])[:500])

    if not skip_report:
        html_path, pdf_path = generate_report(
            query=query,
            max_files=max_files,
            max_tables=max_tables or 10,
            html_only=html_only,
        )

    return search_results, html_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Law table RAG pipeline runner")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--strategy", default="item", choices=["row", "item", "table", "metadata"])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-tables", type=int, default=None)
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        query=args.query,
        skip_ingest=args.skip_ingest,
        skip_report=args.skip_report,
        reset=args.reset,
        strategy=args.strategy,
        max_files=args.max_files,
        max_tables=args.max_tables,
        html_only=args.html_only,
    )


if __name__ == "__main__":
    main()
