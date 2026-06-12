"""CPU-forced wrapper for the integrated RAG chatbot."""

from __future__ import annotations

from collections.abc import Iterator

from rag.chatbot import rag_chat as _rag_chat
from rag.chatbot import rag_chat_stream as _rag_chat_stream
from rag.config import RAG_TOP_K
from rag.schemas import ChatRequest, ChatResponse, SourceDoc


def rag_chat(
    request: ChatRequest,
    *,
    text_top_k: int = RAG_TOP_K,
    table_top_k: int = RAG_TOP_K,
) -> ChatResponse:
    return _rag_chat(
        request,
        text_top_k=text_top_k,
        table_top_k=table_top_k,
        cpu=True,
    )


def rag_chat_stream(
    request: ChatRequest,
    *,
    text_top_k: int = RAG_TOP_K,
    table_top_k: int = RAG_TOP_K,
) -> Iterator[tuple[str, list[SourceDoc]]]:
    return _rag_chat_stream(
        request,
        text_top_k=text_top_k,
        table_top_k=table_top_k,
        cpu=True,
    )
