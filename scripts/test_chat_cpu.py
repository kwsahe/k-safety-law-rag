"""Interactive integrated RAG chatbot with CPU mode for local Ollama fallback."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chatbot_cpu import rag_chat_stream  # noqa: E402
from rag.config import LLM_PROVIDER  # noqa: E402
from rag.schemas import ChatRequest  # noqa: E402


def print_separator(char: str = "-", width: int = 70) -> None:
    print(char * width)


def print_sources(sources) -> None:
    print("\n[참고 근거]")
    for index, doc in enumerate(sources, start=1):
        metadata = doc.metadata
        source_type = metadata.get("source_type", "")
        law_name = metadata.get("law_name", "")
        article = metadata.get("article", "")
        page = metadata.get("page", "")
        score = metadata.get("score", 0.0)
        label = f"{law_name} {article}".strip()
        print(f"  {index}. [{source_type}] {label} p.{page} score={score}")


def main() -> None:
    print_separator("=")
    print("  Focus-Report Integrated RAG Chat [text + table / CPU]")
    print("  종료: exit 또는 Ctrl+C")
    print_separator("=")

    while True:
        try:
            question = input("\n질문: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break

        if not question:
            continue
        if question.lower() == "exit":
            print("종료합니다.")
            break

        start = time.time()
        sources = []
        print("\n[답변]")
        try:
            for token, srcs in rag_chat_stream(ChatRequest(question=question)):
                print(token, end="", flush=True)
                sources = srcs
        except RuntimeError as exc:
            print(f"\n오류: {exc}")
            if LLM_PROVIDER == "remote_openai":
                print("Colab EXAONE 서버 URL(.env의 LLM_API_BASE)이 열려 있는지 확인하세요.")
            else:
                print("Ollama 서버가 실행 중인지 확인하세요. 예: ollama serve")
            continue

        elapsed_ms = int((time.time() - start) * 1000)
        print_sources(sources)
        print(f"\n응답 시간: {elapsed_ms}ms")
        print_separator()


if __name__ == "__main__":
    main()
