"""Backward-compatible wrapper for the interactive RAG CLI.

Prefer running ``python cli.py`` from the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.cli import main


if __name__ == "__main__":
    main()
