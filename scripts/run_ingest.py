"""CLI wrapper for text-law ingestion."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.ingest import main as ingest_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest law PDFs into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="reset the collection before ingest")
    parser.add_argument("--dir", type=str, default=None, help="law PDF directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    laws_dir = Path(args.dir).resolve() if args.dir else None

    start = time.time()
    ingest_main(reset=args.reset, laws_dir=laws_dir)
    elapsed = time.time() - start
    print(f"[done] elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
