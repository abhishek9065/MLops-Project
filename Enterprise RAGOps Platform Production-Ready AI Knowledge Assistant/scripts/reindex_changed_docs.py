from __future__ import annotations

from pathlib import Path

from scripts.ingest_documents import ingest_path
from app.vector_store import SQLiteVectorStore


def main() -> None:
    # Phase 1 upsert logic already skips unchanged files with the same filename and SHA-256.
    ingest_path(Path("data/raw"), SQLiteVectorStore())


if __name__ == "__main__":
    main()

