"""Table extraction utilities for Korean law PDFs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pdfplumber

from rag.config import LAWS_DIR, METADATA_PATH
from rag.table_chunking import clean_table

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PDFPLUMBER_TABLE_SETTINGS: dict[str, Any] = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 3,
    "text_tolerance": 2,
}

EXPOSURE_LIMIT_COLUMNS = [
    "유해인자",
    "TWA_ppm",
    "TWA_mg_m3",
    "STEL_ppm",
    "STEL_mg_m3",
]

EXPOSURE_LIMIT_COLUMNS_WITH_DETAIL = [
    "유해인자",
    "세부구분",
    "TWA_ppm",
    "TWA_mg_m3",
    "STEL_ppm",
    "STEL_mg_m3",
]


def load_law_metadata() -> list[dict[str, Any]]:
    """Load law metadata, or derive minimal metadata by scanning PDFs."""
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data

    entries: list[dict[str, Any]] = []
    for pdf_path in sorted(LAWS_DIR.glob("*/*.pdf")):
        entries.append(
            {
                "law_name": pdf_path.stem,
                "law_short": pdf_path.stem,
                "category": "unknown",
                "pdf_file": pdf_path.name,
                "folder": pdf_path.parent.name,
            }
        )
    return entries


def resolve_law_pdf(entry: dict[str, Any]) -> Path:
    """Resolve a metadata entry to an absolute PDF path."""
    return LAWS_DIR / str(entry["folder"]) / str(entry["pdf_file"])


def iter_law_pdfs(max_files: int | None = None) -> Iterable[tuple[dict[str, Any], Path]]:
    """Yield metadata entries with existing PDF paths."""
    count = 0
    for entry in load_law_metadata():
        pdf_path = resolve_law_pdf(entry)
        if not pdf_path.exists():
            print(f"[skip] missing PDF: {pdf_path}")
            continue

        yield entry, pdf_path
        count += 1
        if max_files is not None and count >= max_files:
            break


def raw_table_to_df(raw_table: list[list[Any]]) -> pd.DataFrame:
    """Convert a raw pdfplumber table into a normalized DataFrame."""
    if not raw_table:
        return pd.DataFrame()

    normalized_rows = normalize_raw_rows(raw_table)
    exposure_df = exposure_limit_table_to_df(normalized_rows)
    if exposure_df is not None:
        return exposure_df

    # If the first cell starts with a numbered item pattern (e.g. "17.") AND
    # the table has <=3 columns (별표5 특별교육 등), treat all rows as data.
    # Exclude wide tables (4+ columns) like 별표19 노출기준(5~6열) which have
    # real header rows and named columns (TWA_ppm, STEL_ppm, etc.).
    first_cell = normalized_rows[0][0] if normalized_rows[0] else ""
    num_cols = len(normalized_rows[0])
    if re.match(r"^\d+\.", first_cell) and num_cols <= 3:
        header = [f"col_{idx}" for idx in range(num_cols)]
        body = normalized_rows
    else:
        header = [
            cell if cell else f"col_{idx}"
            for idx, cell in enumerate(normalized_rows[0])
        ]
        body = normalized_rows[1:]

    if not body:
        return pd.DataFrame()

    return clean_table(pd.DataFrame(body, columns=header))


def normalize_raw_rows(raw_table: list[list[Any]]) -> list[list[str]]:
    """Pad ragged pdfplumber rows and normalize whitespace."""
    max_cols = max(len(row) for row in raw_table)
    normalized: list[list[str]] = []
    for row in raw_table:
        padded = list(row) + [""] * (max_cols - len(row))
        normalized.append([normalize_cell(cell) for cell in padded])
    return normalized


def normalize_cell(cell: Any) -> str:
    if cell in (None, ""):
        return ""
    return " ".join(str(cell).split())


def exposure_limit_table_to_df(rows: list[list[str]]) -> pd.DataFrame | None:
    """Normalize [별표 19] exposure-limit tables, including continuation pages."""
    if not rows:
        return None

    if is_exposure_limit_header_table(rows):
        columns = (
            EXPOSURE_LIMIT_COLUMNS_WITH_DETAIL
            if len(rows[0]) >= len(EXPOSURE_LIMIT_COLUMNS_WITH_DETAIL)
            else EXPOSURE_LIMIT_COLUMNS
        )
        return exposure_rows_to_df(rows[3:], columns)

    if is_exposure_limit_continuation_table(rows):
        columns = (
            EXPOSURE_LIMIT_COLUMNS_WITH_DETAIL
            if len(rows[0]) >= len(EXPOSURE_LIMIT_COLUMNS_WITH_DETAIL)
            else EXPOSURE_LIMIT_COLUMNS
        )
        return exposure_rows_to_df(rows, columns)

    return None


def is_exposure_limit_header_table(rows: list[list[str]]) -> bool:
    header_text = " ".join(" ".join(row) for row in rows[:3])
    return (
        "유해인자" in header_text
        and "허용기준" in header_text
        and "시간가중평균값" in header_text
        and "단시간 노출값" in header_text
    )


def is_exposure_limit_continuation_table(rows: list[list[str]]) -> bool:
    if len(rows[0]) not in (5, 6):
        return False

    first_cells = [row[0] for row in rows[:8] if row]
    numbered = sum(1 for cell in first_cells if re.match(r"^\d+\.", cell))
    if numbered < 2:
        return False

    table_text = " ".join(" ".join(row) for row in rows[:8])
    has_chemical_names = bool(re.search(r"\([A-Za-z]", table_text)) or "화합물" in table_text
    numeric_limit_cells = sum(
        1
        for row in rows[:8]
        for cell in row[1:]
        if re.search(r"\d", cell)
    )
    return has_chemical_names and numeric_limit_cells >= 2


def exposure_rows_to_df(rows: list[list[str]], columns: list[str]) -> pd.DataFrame:
    normalized = [
        row[: len(columns)] + [""] * max(0, len(columns) - len(row))
        for row in rows
    ]
    df = clean_table(pd.DataFrame(normalized, columns=columns))
    if "세부구분" in df.columns and "유해인자" in df.columns:
        df["유해인자"] = df["유해인자"].replace("", pd.NA).ffill().fillna("")
    return df


def extract_tables_from_pdf(
    pdf_path: Path,
    entry: dict[str, Any],
    max_tables: int | None = None,
) -> list[tuple[pd.DataFrame, dict[str, Any]]]:
    """Extract tables from one law PDF with pdfplumber."""
    extracted: list[tuple[pd.DataFrame, dict[str, Any]]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                raw_tables = page.extract_tables(table_settings=PDFPLUMBER_TABLE_SETTINGS) or []
            except Exception:
                raw_tables = page.extract_tables() or []

            for table_index, raw_table in enumerate(raw_tables):
                df = raw_table_to_df(raw_table)
                if df.empty:
                    continue

                metadata = {
                    "law_name": entry.get("law_name", pdf_path.stem),
                    "law_short": entry.get("law_short", pdf_path.stem),
                    "category": entry.get("category", "unknown"),
                    "pdf_file": pdf_path.name,
                    "page": page_num,
                    "table_index": table_index,
                    "source": str(pdf_path.relative_to(LAWS_DIR.parent)),
                }
                extracted.append((df, metadata))

                if max_tables is not None and len(extracted) >= max_tables:
                    return extracted

    return extracted


def extract_all_tables(
    max_files: int | None = None,
    max_tables: int | None = None,
) -> list[tuple[pd.DataFrame, dict[str, Any]]]:
    """Extract tables from all configured law PDFs."""
    result: list[tuple[pd.DataFrame, dict[str, Any]]] = []

    for entry, pdf_path in iter_law_pdfs(max_files=max_files):
        remaining = None if max_tables is None else max_tables - len(result)
        if remaining is not None and remaining <= 0:
            break

        tables = extract_tables_from_pdf(pdf_path, entry, max_tables=remaining)
        print(f"[extract] {pdf_path.name}: {len(tables)} tables")
        result.extend(tables)

    print(f"[extract] total tables: {len(result)}")
    return result


def extract_tables_with_camelot(pdf_path: Path, flavor: str = "lattice") -> list[pd.DataFrame]:
    """Optional Camelot extraction helper for manual comparison."""
    try:
        import camelot
    except ImportError as exc:
        raise RuntimeError("camelot-py is not installed") from exc

    tables = camelot.read_pdf(str(pdf_path), pages="all", flavor=flavor)
    result: list[pd.DataFrame] = []
    for table in tables:
        df = clean_table(table.df)
        df.attrs["accuracy"] = table.accuracy
        df.attrs["whitespace"] = table.whitespace
        df.attrs["page"] = table.page
        result.append(df)
    return result


def extract_tables_with_pymupdf(pdf_path: Path) -> list[pd.DataFrame]:
    """Optional PyMuPDF extraction helper for manual comparison."""
    import fitz

    result: list[pd.DataFrame] = []
    doc = fitz.open(str(pdf_path))
    try:
        for page_num, page in enumerate(doc, start=1):
            if not hasattr(page, "find_tables"):
                continue
            finder = page.find_tables()
            for table_index, table in enumerate(finder.tables):
                df = clean_table(table.to_pandas())
                df.attrs["page"] = page_num
                df.attrs["table_index"] = table_index
                result.append(df)
    finally:
        doc.close()
    return result


def preview_tables(max_files: int = 1, max_tables: int = 3) -> None:
    """Print a lightweight extraction preview without embedding."""
    tables = extract_all_tables(max_files=max_files, max_tables=max_tables)
    for index, (df, metadata) in enumerate(tables, start=1):
        print("\n" + "=" * 80)
        print(
            f"[{index}] {metadata['law_name']} "
            f"p.{metadata['page']} table#{metadata['table_index'] + 1} "
            f"shape={df.shape}"
        )
        print(df.head(8).to_string(index=False, max_cols=8))
