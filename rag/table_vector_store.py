"""ChromaDB wrapper for law table chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from rag.config import TABLE_CHROMA_COLLECTION, TABLE_CHROMA_PATH


def _disable_chroma_telemetry() -> None:
    """Silence ChromaDB/PostHog telemetry errors in local runs."""
    try:
        from chromadb.telemetry.product.posthog import Posthog

        Posthog.capture = lambda self, event: None
    except Exception:
        pass


class TableVectorStore:
    """Small ChromaDB facade dedicated to extracted law tables."""

    def __init__(
        self,
        persist_path: Path = TABLE_CHROMA_PATH,
        collection_name: str = TABLE_CHROMA_COLLECTION,
    ) -> None:
        _disable_chroma_telemetry()
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert chunks so repeated law-data tests are idempotent."""
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate only the table collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


_instance: TableVectorStore | None = None


def get_table_vector_store() -> TableVectorStore:
    global _instance
    if _instance is None:
        _instance = TableVectorStore()
    return _instance
