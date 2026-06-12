"""Generate HTML/PDF reports from extracted law tables."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from rag.config import TABLE_REPORT_OUTPUT_DIR, TEMPLATE_DIR
from rag.table_extraction import extract_all_tables
from rag.table_retriever import search_table_chunks

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_table_sections(
    max_files: int | None = None,
    max_tables: int | None = None,
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for df, metadata in extract_all_tables(max_files=max_files, max_tables=max_tables):
        sections.append(
            {
                "name": (
                    f"{metadata['law_name']} table#{metadata['table_index'] + 1} "
                    f"(p.{metadata['page']})"
                ),
                "meta": f"category={metadata['category']} | file={metadata['pdf_file']}",
                "columns": [str(col) for col in df.columns],
                "rows": df.values.tolist(),
                "law_name": metadata["law_name"],
            }
        )
    return sections


def build_summary(sections: list[dict[str, Any]]) -> list[dict[str, str]]:
    total_rows = sum(len(section["rows"]) for section in sections)
    law_count = len({section["law_name"] for section in sections})
    return [
        {"label": "laws", "value": str(law_count)},
        {"label": "tables", "value": str(len(sections))},
        {"label": "rows", "value": str(total_rows)},
        {"label": "generated_at", "value": datetime.now().strftime("%Y-%m-%d %H:%M")},
    ]


def render_html(
    sections: list[dict[str, Any]],
    summary: list[dict[str, str]],
    query: str | None = None,
    search_results: list[dict[str, Any]] | None = None,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("table_report.html")
    css = (TEMPLATE_DIR / "table_report.css").read_text(encoding="utf-8")
    return template.render(
        css=css,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        sections=sections,
        summary=summary,
        query=query,
        search_results=search_results or [],
    )


def save_html(html: str, stem: str, output_dir: Path = TABLE_REPORT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.html"
    path.write_text(html, encoding="utf-8")
    print(f"[html] {path}")
    return path


def save_pdf(html: str, stem: str, output_dir: Path = TABLE_REPORT_OUTPUT_DIR) -> Path:
    try:
        from weasyprint import HTML
    except ImportError as exc:
        raise RuntimeError("weasyprint is not installed") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.pdf"
    HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf(str(path))
    print(f"[pdf] {path}")
    return path


def generate_report(
    query: str | None = None,
    top_k: int = 5,
    max_files: int | None = None,
    max_tables: int | None = 10,
    html_only: bool = False,
) -> tuple[Path, Path | None]:
    sections = load_table_sections(max_files=max_files, max_tables=max_tables)
    summary = build_summary(sections)
    search_results = search_table_chunks(query, n_results=top_k) if query else []

    stem = f"table_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    html = render_html(sections, summary, query, search_results)
    html_path = save_html(html, stem)
    pdf_path = None if html_only else save_pdf(html, stem)
    return html_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate law table RAG report")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-tables", type=int, default=10)
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()

    generate_report(
        query=args.query,
        top_k=args.top_k,
        max_files=args.max_files,
        max_tables=args.max_tables,
        html_only=args.html_only,
    )


if __name__ == "__main__":
    main()
