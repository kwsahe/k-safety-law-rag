"""PDF law text ingestion pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, LAWS_DIR
from rag.embedder import get_embedder
from rag.vector_store import get_vector_store

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logger = logging.getLogger(__name__)


def _load_law_name_map() -> dict[str, str]:
    """Build pdf_file -> law_name lookup from _metadata.json."""
    try:
        from rag.config import METADATA_PATH
        data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return {item["pdf_file"]: item["law_name"] for item in data}
    except Exception:
        return {}


_LAW_NAME_MAP: dict[str, str] = _load_law_name_map()


def load_pdf_files(laws_dir: Path | None = None) -> list[Path]:
    """Return all PDFs under the configured law directory."""
    target = laws_dir if laws_dir is not None else LAWS_DIR
    pdf_paths = sorted(target.glob("**/*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under {target}")
    logger.info("%s PDF files found under %s", len(pdf_paths), target)
    return pdf_paths


def load_documents(pdf_paths: list[Path]) -> list[Any]:
    """Load PDFs into LangChain documents."""
    documents: list[Any] = []
    for pdf_path in pdf_paths:
        print(f"[load] {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    print(f"[load] pages={len(documents)}")
    return documents


def chunk_documents(documents: list[Any]) -> list[Any]:
    """Split law documents by article-friendly separators."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n제", "\n\n", "\n", " "],
        keep_separator=True,  # preserve '제' so article numbers stay intact
    )
    chunks = splitter.split_documents(documents)
    print(f"[chunk] chunks={len(chunks)}")
    return chunks


def extract_article_number(text: str) -> str:
    """Extract Korean article labels such as '제6조' or '제6조의2'."""
    match = re.search(r"제\s*\d+\s*조(?:\s*의\s*\d+)?", text)
    return re.sub(r"\s+", "", match.group(0)) if match else ""


def _chunk_id(source: str, page: int, index: int) -> str:
    """Stable deterministic ID from source + page + index."""
    key = f"{source}::p{page}::i{index}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _chunk_metadata(chunk: Any) -> dict[str, Any]:
    source_file = os.path.basename(chunk.metadata.get("source", "unknown"))
    law_name = _LAW_NAME_MAP.get(source_file, Path(source_file).stem)
    return {
        "source": source_file,
        "page": int(chunk.metadata.get("page", 0)),
        "law_name": law_name,
        "article": extract_article_number(chunk.page_content),
    }


def embed_and_store(chunks: list[Any], reset: bool = False, batch_size: int = 100) -> int:
    """Embed chunks and upsert them into the text-law ChromaDB collection."""
    embedder = get_embedder()
    vector_store = get_vector_store()

    if reset:
        vector_store.reset()

    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk.page_content for chunk in batch]
        embeddings = embedder.encode_batch(texts, batch_size=batch_size)
        metadatas = [_chunk_metadata(chunk) for chunk in batch]
        ids = [
            _chunk_id(
                meta["source"],
                meta["page"],
                start + index,
            )
            for index, meta in enumerate(metadatas)
        ]

        vector_store.add_documents(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"[store] {min(start + batch_size, total)}/{total}")

    return total


def main(reset: bool = False, laws_dir: Path | None = None) -> None:
    """Run the full PDF text ingestion pipeline."""
    pdf_paths = load_pdf_files(laws_dir)
    documents = load_documents(pdf_paths)
    chunks = chunk_documents(documents)
    total = embed_and_store(chunks, reset=reset)
    print(f"[done] stored_chunks={total}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Ingest law PDFs into text ChromaDB")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--dir", type=str, default=None)
    args = parser.parse_args()

    laws_dir = Path(args.dir).resolve() if args.dir else None
    main(reset=args.reset, laws_dir=laws_dir)


if __name__ == "__main__":
    cli()
