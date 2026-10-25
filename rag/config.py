"""Project-wide configuration for text and table RAG modules."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR: Path = Path(__file__).parent.parent.resolve()

load_dotenv(ROOT_DIR / ".env")


def _resolve_path(env_name: str, default: Path, must_exist: bool = False) -> Path:
    raw = os.getenv(env_name)
    if not raw:
        return default

    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    candidate = candidate.resolve()

    if must_exist and not candidate.exists():
        return default
    return candidate


# Base paths
RAG_DIR: Path = ROOT_DIR / "rag"
DATA_DIR: Path = ROOT_DIR / "data"
LAWS_DIR: Path = _resolve_path("LAWS_DIR", DATA_DIR / "laws", must_exist=True)
METADATA_PATH: Path = LAWS_DIR / "_metadata.json"

# Text law RAG storage
CHROMA_PATH: Path = _resolve_path("CHROMA_PATH", ROOT_DIR / "chroma_db")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "korean_safety_laws")

# Table law RAG storage
TABLE_CHROMA_PATH: Path = _resolve_path(
    "TABLE_CHROMA_PATH",
    ROOT_DIR / "chroma_db_tables",
)
TABLE_CHROMA_COLLECTION: str = os.getenv("TABLE_CHROMA_COLLECTION", "law_tables")
RAG_TABLE_BATCH_SIZE: int = int(os.getenv("RAG_TABLE_BATCH_SIZE", "64"))

# Report output/templates
TEMPLATE_DIR: Path = ROOT_DIR / "templates"
TABLE_REPORT_OUTPUT_DIR: Path = ROOT_DIR / "output" / "table_reports"

# Models
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "remote_openai")
# LLM_PROVIDER=anthropic switches to the Claude API (key: ANTHROPIC_API_KEY);
# the Colab/Ollama model settings stay in LLM_MODEL untouched.
LLM_MODEL: str = (
    os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
    if LLM_PROVIDER == "anthropic"
    else os.getenv("LLM_MODEL", "LGAI-EXAONE/EXAONE-4.0-32B")
)
ANTHROPIC_EFFORT: str = os.getenv("ANTHROPIC_EFFORT", "medium")
# Colab servers the admin dashboard can check (notebooks/RAG_*_colab_*_server.ipynb).
EXAONE_API_BASE: str = os.getenv("EXAONE_API_BASE", "")
EXAONE_MODEL: str = os.getenv("EXAONE_MODEL", "LGAI-EXAONE/EXAONE-4.0-32B")
QWEN_API_BASE: str = os.getenv("QWEN_API_BASE", "")
QWEN_MODEL: str = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-14B-Instruct")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_API_BASE: str = os.getenv("LLM_API_BASE", "")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")

# Retrieval/chunking parameters
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
RAG_CONTEXT_TABLE_K: int = int(os.getenv("RAG_CONTEXT_TABLE_K", "3"))
RAG_CONTEXT_TEXT_K: int = int(os.getenv("RAG_CONTEXT_TEXT_K", "2"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
