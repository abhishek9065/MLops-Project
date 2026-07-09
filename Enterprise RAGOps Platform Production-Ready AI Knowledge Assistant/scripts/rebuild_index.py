from __future__ import annotations

from pathlib import Path

from scripts.ingest_documents import ingest_path
from app.vector_store import SQLiteVectorStore


def main() -> None:
    db_path = Path("data/processed/ragops.db")
    if db_path.exists():
        db_path.unlink()
    ingest_path(Path("data/raw"), SQLiteVectorStore(db_path))


if __name__ == "__main__":
    main()

