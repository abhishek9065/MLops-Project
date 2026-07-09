from __future__ import annotations

import argparse
from pathlib import Path

from scripts.ingest_documents import ingest_path
from app.vector_store import SQLiteVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete and rebuild the local SQLite vector index.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--db", type=Path, default=Path("data/processed/ragops.db"))
    args = parser.parse_args()

    db_path = args.db
    if db_path.exists():
        db_path.unlink()
    ingest_path(args.source_dir, SQLiteVectorStore(db_path))


if __name__ == "__main__":
    main()
