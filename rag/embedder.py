"""Shared bge-m3 embedding wrapper."""

from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

from rag.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self) -> None:
        print(f"[Embedder] loading model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("[Embedder] model loaded.")

    def encode(self, text: str) -> List[float]:
        """Encode one string into a Python list vector."""
        return self.model.encode(text).tolist()

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Encode multiple strings with sentence-transformers batching."""
        return self.model.encode(texts, batch_size=batch_size).tolist()


_embedder_instance: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the shared embedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
