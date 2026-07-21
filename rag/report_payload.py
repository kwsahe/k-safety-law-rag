"""Stable page 7 and page 12 payloads for v1 report integration."""

from __future__ import annotations

import re
from typing import Any

from rag.schemas import SourceDoc


def _source_label(source: SourceDoc) -> str:
    metadata = source.metadata
    law_name = str(metadata.get("law_name") or "법령").replace("_", " ")
    reference = str(metadata.get("annex") or metadata.get("article") or "").strip()
    page = str(metadata.get("citation_page") or metadata.get("page") or "").strip()
    page_suffix = f", p.{page}" if page and page != "0" else ""
    return f"{law_name} {reference}{page_suffix}".strip()


def _violation_lines(answer: str) -> list[str]:
    lines: list[str] = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        if not any(term in line for term in ("위반", "미실시", "미흡", "미준수", "미사용", "해체", "근거:")):
            continue
        cleaned = re.sub(r"^-\s*", "", line).strip()
        if cleaned and cleaned not in lines:
            lines.append(cleaned)
    return lines[:9]


def build_report_pages(answer: str, sources: list[SourceDoc]) -> list[dict[str, Any]]:
    """Return the fixed two-page payload expected by the report service."""
    labels: list[str] = []
    for source in sources:
        label = _source_label(source)
        if label and label not in labels:
            labels.append(label)

    violations = _violation_lines(answer)
    items = []
    for index, violation in enumerate(violations, start=1):
        items.append(
            {
                "marker": f"{index}",
                "law_item": labels[index - 1] if index <= len(labels) else "답변 본문 근거 참조",
                "violation": violation,
            }
        )

    return [
        {
            "page": 7,
            "data": {
                "legal_violations": {
                    "title": f"법령 위반 사항 ({len(items)}건)",
                    "items": items,
                },
                "citation_count": len(labels),
                "citations": labels,
            },
        },
        {
            "page": 12,
            "data": {
                "accident_final_evaluation": answer,
                "summary": next((line.strip() for line in answer.splitlines() if line.strip()), ""),
            },
        },
    ]
