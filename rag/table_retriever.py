"""Ingest and search law table chunks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

from tqdm import tqdm

from rag.config import (
    RAG_TABLE_BATCH_SIZE,
    RAG_TOP_K,
    TABLE_CHROMA_COLLECTION,
    TABLE_CHROMA_PATH,
)
from rag.embedder import get_embedder
from rag.table_chunking import TableChunk, chunk_tables
from rag.table_extraction import extract_all_tables, preview_tables
from rag.table_vector_store import get_table_vector_store

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convert metadata values into ChromaDB-compatible scalars."""
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif value is None:
            clean[key] = ""
        else:
            clean[key] = json.dumps(value, ensure_ascii=False, default=str)
    return clean


def embed_and_store_chunks(
    chunks: list[TableChunk],
    batch_size: int = RAG_TABLE_BATCH_SIZE,
) -> int:
    """Embed and upsert chunks into the table vector store."""
    if not chunks:
        return 0

    embedder = get_embedder()
    vector_store = get_table_vector_store()
    written = 0

    for start in tqdm(range(0, len(chunks), batch_size), desc="Embedding table chunks"):
        batch = chunks[start : start + batch_size]
        texts = [chunk.text for chunk in batch]
        embeddings = embedder.encode_batch(texts, batch_size=batch_size)
        vector_store.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[sanitize_metadata(chunk.metadata) for chunk in batch],
        )
        written += len(batch)

    return written


def ingest_tables(
    strategy: str = "row",
    reset: bool = False,
    max_files: int | None = None,
    max_tables: int | None = None,
    batch_size: int = RAG_TABLE_BATCH_SIZE,
) -> int:
    """Extract law tables, chunk them, and store them in ChromaDB."""
    vector_store = get_table_vector_store()
    if reset:
        vector_store.reset()

    tables = extract_all_tables(max_files=max_files, max_tables=max_tables)
    chunks = chunk_tables(tables, strategy=strategy)
    print(f"[chunk] strategy={strategy}, chunks={len(chunks)}")

    written = embed_and_store_chunks(chunks, batch_size=batch_size)
    print(
        "[ingest] "
        f"written={written}, collection={TABLE_CHROMA_COLLECTION}, "
        f"path={TABLE_CHROMA_PATH}, total={vector_store.count()}"
    )
    return written


def search_table_chunks(query: str, n_results: int = RAG_TOP_K) -> list[dict[str, Any]]:
    """Search indexed table chunks."""
    embedder = get_embedder()
    vector_store = get_table_vector_store()
    query_vector = embedder.encode(query)
    results = vector_store.query(query_vector, n_results=n_results)

    hits: list[dict[str, Any]] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, distance in zip(docs, metas, distances):
        hits.append(
            {
                "text": doc,
                "score": round(1 - float(distance), 4),
                "metadata": meta,
            }
        )
    return hits


def get_all_table_chunks() -> list[dict[str, Any]]:
    """Return all stored table chunks for lightweight lexical rescoring."""
    vector_store = get_table_vector_store()
    results = vector_store.collection.get(include=["documents", "metadatas"])
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []
    return [
        {
            "text": str(doc),
            "metadata": meta or {},
            "score": 0.0,
        }
        for doc, meta in zip(docs, metas)
    ]


def search_table_chunks_lexical(query: str, n_results: int = RAG_TOP_K) -> list[dict[str, Any]]:
    """Search table chunks by exact terms to complement embedding search."""
    terms = extract_query_terms(query)
    if not terms:
        return []

    hits: list[dict[str, Any]] = []
    for chunk in get_all_table_chunks():
        text = str(chunk["text"])
        score = lexical_score(query, text, terms)
        if score <= 0:
            continue
        hits.append({**chunk, "score": round(score, 4)})

    hits.sort(key=lambda hit: hit["score"], reverse=True)
    return hits[:n_results]


def find_neighbor_table_chunks(
    metadata: dict[str, Any],
    before: int = 0,
    after: int = 2,
) -> list[dict[str, Any]]:
    """Find nearby row chunks from the same extracted table."""
    row_index = metadata.get("row_index")
    table_index = metadata.get("table_index")
    page = metadata.get("page")
    source = metadata.get("source") or metadata.get("pdf_file")
    if row_index is None or table_index is None or page is None:
        return []

    try:
        row_index_int = int(row_index)
    except (TypeError, ValueError):
        return []

    start = row_index_int - before
    end = row_index_int + after
    neighbors: list[dict[str, Any]] = []
    for chunk in get_all_table_chunks():
        meta = chunk["metadata"]
        chunk_source = meta.get("source") or meta.get("pdf_file")
        if chunk_source != source:
            continue
        if meta.get("page") != page or meta.get("table_index") != table_index:
            continue
        try:
            candidate_row = int(meta.get("row_index"))
        except (TypeError, ValueError):
            continue
        if start <= candidate_row <= end:
            neighbors.append({**chunk, "score": max(float(chunk.get("score", 0.0)), 0.0001)})

    neighbors.sort(key=lambda hit: int(hit["metadata"].get("row_index", 0)))
    return neighbors


def extract_query_terms(query: str) -> list[str]:
    """Extract stable Korean/English terms for exact table lookup."""
    lower = query.lower()
    compact = re.sub(r"\s+", "", query)

    aliases = {
        "카드뮴": ["카드뮴", "cadmium"],
        "cadmium": ["카드뮴", "cadmium"],
        "6가크롬": ["6가크롬", "chromium vi", "chromium"],
        "chromium": ["6가크롬", "chromium"],
        "석면": ["석면", "asbestos"],
        "asbestos": ["석면", "asbestos"],
        "벤젠": ["벤젠", "benzene"],
        "benzene": ["벤젠", "benzene"],
        "굴착": ["굴착", "지반 굴착", "굴착면", "2미터", "특별교육"],
        "굴착면": ["굴착", "지반 굴착", "굴착면", "2미터", "특별교육"],
        "지반굴착": ["굴착", "지반 굴착", "굴착면", "2미터", "특별교육"],
        "특별교육": ["특별교육", "교육내용", "작업항목", "굴착"],
        "크레인": ["크레인", "인양", "화물", "신호방법", "작업항목"],
        "인양": ["크레인", "인양", "화물", "신호방법", "작업항목"],
        "철골": ["골조", "금속제", "조립", "해체", "5미터", "작업항목"],
        "골조": ["골조", "금속제", "조립", "해체", "5미터", "작업항목"],
        "금속": ["골조", "금속제", "조립", "해체", "5미터", "작업항목"],
    }

    terms: list[str] = []
    for key, values in aliases.items():
        if key in lower or key in compact:
            terms.extend(values)

    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        normalized = term.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(term)
    return unique


def lexical_score(query: str, text: str, terms: list[str]) -> float:
    normalized_text = text.lower()
    compact_text = re.sub(r"\s+", "", normalized_text)
    score = 0.0
    for term in terms:
        normalized_term = term.lower()
        compact_term = re.sub(r"\s+", "", normalized_term)
        if normalized_term in normalized_text or compact_term in compact_text:
            score += 1.0

    if score <= 0:
        return 0.0

    exposure_query = any(
        term in query.lower() or term in query
        for term in ("twa", "stel", "허용기준", "노출기준", "mg/㎥", "mg/m3", "ppm")
    )
    if exposure_query:
        if not has_exposure_limit_signal(text):
            return 0.0
        score += 2.0

    compact_query = re.sub(r"\s+", "", query)
    compact_text = re.sub(r"\s+", "", text)
    excavation_education_query = (
        any(term in compact_query for term in ("굴착", "지반굴착", "굴착면"))
        and any(term in compact_query for term in ("특별교육", "교육", "미이수", "미실시", "위반"))
    )
    if excavation_education_query:
        if "[작업항목]19." in compact_text or "굴착면의높이가2미터이상인지반굴착작업" in compact_text:
            score += 4.0
        elif "[작업항목]21." in compact_text:
            if "터널" not in compact_query:
                score -= 2.0
        elif "[작업항목]22." in compact_text:
            if not any(term in compact_query for term in ("암석", "발파", "폭발")):
                score -= 3.0

    crane_query = any(term in compact_query for term in ("크레인", "인양", "양중"))
    if crane_query and ("[작업항목]14." in compact_text or "크레인을사용하는작업" in compact_text):
        score += 4.0

    steel_query = any(term in compact_query for term in ("철골", "골조", "금속제", "금속", "15층", "고층"))
    if steel_query and ("[작업항목]27." in compact_text or "건축물의골조" in compact_text):
        score += 4.0

    return normalize_lexical_score(score, exact=score >= 4.0)


def normalize_lexical_score(score: float, *, exact: bool = False) -> float:
    """Map lexical term counts into the same 0~1 display range as vector scores."""
    if score <= 0:
        return 0.0
    if exact:
        return 0.96
    return round(min(0.9, 0.45 + score * 0.08), 4)


def has_exposure_limit_signal(text: str) -> bool:
    return any(
        signal in text
        for signal in (
            "TWA_",
            "STEL_",
            "허용기준",
            "시간가중평균값",
            "단시간 노출값",
            "col_2:",
            "col_3:",
            "col_4:",
            "col_5:",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Law table RAG pipeline")
    parser.add_argument("--ingest", action="store_true", help="extract, embed, and store tables")
    parser.add_argument("--reset", action="store_true", help="reset the table collection before ingest")
    parser.add_argument("--extract-only", action="store_true", help="preview table extraction only")
    parser.add_argument("--strategy", default="item", choices=["row", "item", "table", "metadata"])
    parser.add_argument("--query", type=str, help="search query")
    parser.add_argument("--top-k", type=int, default=RAG_TOP_K)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-tables", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=RAG_TABLE_BATCH_SIZE)
    args = parser.parse_args()

    if args.extract_only:
        preview_tables(max_files=args.max_files or 1, max_tables=args.max_tables or 3)

    if args.ingest:
        ingest_tables(
            strategy=args.strategy,
            reset=args.reset,
            max_files=args.max_files,
            max_tables=args.max_tables,
            batch_size=args.batch_size,
        )

    if args.query:
        hits = search_table_chunks(args.query, n_results=args.top_k)
        print(f"\nquery: {args.query}\n" + "=" * 80)
        for index, hit in enumerate(hits, start=1):
            metadata = hit["metadata"]
            print(
                f"\n[{index}] score={hit['score']} "
                f"{metadata.get('law_name', '')} p.{metadata.get('page', '')}"
            )
            print(str(hit["text"])[:500])

    if not (args.extract_only or args.ingest or args.query):
        parser.print_help()


if __name__ == "__main__":
    main()
