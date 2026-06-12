"""Convert extracted tables into embedding-friendly chunks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

_ITEM_NUMBER_PATTERN = re.compile(r"^\d+\.")


@dataclass
class TableChunk:
    """A table-derived text chunk and its source metadata."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = make_chunk_id(self.text, self.metadata)


def make_chunk_id(text: str, metadata: dict[str, Any]) -> str:
    payload = json.dumps(
        {"text": text, "metadata": metadata},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a pdfplumber/Camelot/PyMuPDF table DataFrame."""
    cleaned = df.copy()
    cleaned = cleaned.replace({None: "", "None": ""}).fillna("")
    cleaned = cleaned.map(lambda value: " ".join(str(value).split()))

    cleaned = cleaned.loc[:, [str(col).strip() != "" for col in cleaned.columns]]
    cleaned = cleaned[cleaned.apply(lambda row: any(str(v).strip() for v in row), axis=1)]
    cleaned = cleaned.reset_index(drop=True)

    columns: list[str] = []
    seen: dict[str, int] = {}
    for idx, col in enumerate(cleaned.columns):
        name = " ".join(str(col).split()) or f"col_{idx}"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        columns.append(name)
    cleaned.columns = columns
    return cleaned


def table_to_markdown(df: pd.DataFrame) -> str:
    """Render a small Markdown table without requiring tabulate."""
    cleaned = clean_table(df)
    if cleaned.empty:
        return ""

    headers = [str(col) for col in cleaned.columns]
    rows = [[str(cell) for cell in row] for row in cleaned.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def chunk_by_row(
    df: pd.DataFrame,
    base_metadata: dict[str, Any] | None = None,
) -> list[TableChunk]:
    """Create one chunk per non-empty table row."""
    metadata = base_metadata or {}
    cleaned = clean_table(df)
    chunks: list[TableChunk] = []

    for row_idx, row in cleaned.iterrows():
        parts = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
        text = ", ".join(parts)
        if not text:
            continue

        meta = {**metadata, "chunk_strategy": "row", "row_index": int(row_idx)}
        chunks.append(TableChunk(chunk_id=make_chunk_id(text, meta), text=text, metadata=meta))

    return chunks


def chunk_by_table(
    df: pd.DataFrame,
    base_metadata: dict[str, Any] | None = None,
) -> list[TableChunk]:
    """Create one chunk for the entire table."""
    metadata = base_metadata or {}
    text = table_to_markdown(df)
    if not text:
        return []

    meta = {**metadata, "chunk_strategy": "table", "row_count": int(len(clean_table(df)))}
    return [TableChunk(chunk_id=make_chunk_id(text, meta), text=text, metadata=meta)]


def chunk_with_metadata_separation(
    df: pd.DataFrame,
    base_metadata: dict[str, Any] | None = None,
) -> list[TableChunk]:
    """Keep raw rows in metadata and embed only a compact summary."""
    metadata = base_metadata or {}
    cleaned = clean_table(df)
    if cleaned.empty:
        return []

    columns = [str(col) for col in cleaned.columns]
    sample = cleaned.head(2).to_dict(orient="records")
    text = (
        f"columns: {', '.join(columns)}. "
        f"rows: {len(cleaned)}. "
        f"sample: {json.dumps(sample, ensure_ascii=False)}"
    )
    meta = {
        **metadata,
        "chunk_strategy": "metadata",
        "columns": columns,
        "row_count": int(len(cleaned)),
        "raw_json": cleaned.to_json(orient="records", force_ascii=False),
    }
    return [TableChunk(chunk_id=make_chunk_id(text, meta), text=text, metadata=meta)]


def chunk_by_item(
    df: pd.DataFrame,
    base_metadata: dict[str, Any] | None = None,
) -> list[TableChunk]:
    """Group continuation rows under their numbered item and create one chunk per item.

    When pdfplumber extracts tables with vertically merged cells (e.g. 별표5 특별교육),
    continuation rows have an empty first column. This function forward-fills the item
    number so every education-content row stays attached to its parent item, preventing
    items 17/22 from mixing with item 19 during retrieval.
    Falls back to chunk_by_row when no item-number pattern is found.
    """
    metadata = base_metadata or {}
    cleaned = clean_table(df)
    if cleaned.empty:
        return []

    first_col = cleaned.columns[0]

    # Exposure-limit tables already have meaningful column names (TWA_ppm, STEL_ppm, 유해인자).
    # Their numbered first column (e.g. "12. 벤젠") is a substance name, not a work-type item.
    # Fall back to row strategy so direct_exposure_limit_answer() can parse them correctly.
    EXPOSURE_COLS = {"TWA_ppm", "STEL_ppm", "TWA_mg_m3", "STEL_mg_m3", "유해인자", "허용기준"}
    if EXPOSURE_COLS & set(cleaned.columns):
        return chunk_by_row(df, base_metadata)

    # Detect whether this table has numbered items in the first column.
    has_items = any(
        _ITEM_NUMBER_PATTERN.match(str(val).strip())
        for val in cleaned[first_col]
    )
    if not has_items:
        return chunk_by_row(df, base_metadata)

    # Group rows by item number, forward-filling across empty first-column cells.
    groups: list[tuple[str, int, list[pd.Series]]] = []  # (item_text, row_idx, rows)
    current_item: str | None = None
    current_start: int = 0
    current_rows: list[pd.Series] = []

    for row_idx, row in cleaned.iterrows():
        cell = str(row[first_col]).strip()
        if _ITEM_NUMBER_PATTERN.match(cell):
            if current_item is not None:
                groups.append((current_item, current_start, current_rows))
            current_item = cell
            current_start = int(row_idx)
            current_rows = [row]
        else:
            if current_item is not None:
                current_rows.append(row)
            else:
                # Pre-item rows (table header area) → individual chunks
                parts = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
                text = ", ".join(parts)
                if text:
                    meta = {**metadata, "chunk_strategy": "item", "row_index": int(row_idx)}
                    groups.append(("", int(row_idx), [row]))

    if current_item is not None:
        groups.append((current_item, current_start, current_rows))

    chunks: list[TableChunk] = []
    for item_label, start_idx, rows in groups:
        if not item_label:
            # Pre-item header row
            row = rows[0]
            parts = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
            text = ", ".join(parts)
            if not text:
                continue
            meta = {**metadata, "chunk_strategy": "item", "row_index": start_idx}
            chunks.append(TableChunk(chunk_id=make_chunk_id(text, meta), text=text, metadata=meta))
            continue

        # Combine all sub-rows for this item into one text block.
        # Skip the first column (item number) since it's already in the header.
        # Drop generic "col_X" prefixes so the LLM sees clean content.
        content_parts: list[str] = []
        for row in rows:
            for idx, (col, val) in enumerate(row.items()):
                val_str = str(val).strip()
                if not val_str or idx == 0:
                    continue
                content_parts.append(val_str if col.startswith("col_") else f"{col}: {val_str}")

        if not content_parts:
            continue

        text = f"[작업항목] {item_label}\n[교육내용] " + " ".join(content_parts)
        meta = {
            **metadata,
            "chunk_strategy": "item",
            "row_index": start_idx,
            "item_number": item_label[:40],
        }
        chunks.append(TableChunk(chunk_id=make_chunk_id(text, meta), text=text, metadata=meta))

    return chunks


STRATEGY_MAP: dict[
    str,
    Callable[[pd.DataFrame, dict[str, Any] | None], list[TableChunk]],
] = {
    "row": chunk_by_row,
    "item": chunk_by_item,
    "table": chunk_by_table,
    "metadata": chunk_with_metadata_separation,
}


def chunk_tables(
    tables: list[tuple[pd.DataFrame, dict[str, Any]]],
    strategy: str = "row",
) -> list[TableChunk]:
    if strategy not in STRATEGY_MAP:
        choices = ", ".join(sorted(STRATEGY_MAP))
        raise ValueError(f"Unknown chunk strategy: {strategy}. choices: {choices}")

    chunk_fn = STRATEGY_MAP[strategy]
    all_chunks: list[TableChunk] = []
    for df, metadata in tables:
        all_chunks.extend(chunk_fn(df, metadata))
    return all_chunks
