from __future__ import annotations

import argparse
from pathlib import Path

from app.document_loader import chunk_text, load_document
from app.vector_store import SQLiteVectorStore


def ingest_path(path: Path, store: SQLiteVectorStore) -> None:
    if path.is_dir():
        files = [file for file in path.rglob("*") if file.is_file()]
    else:
        files = [path]

    for file_path in files:
        try:
            document = load_document(file_path)
            chunks = chunk_text(document.text)
            stored_document, stored_chunks = store.upsert_document(document, chunks)
            print(f"Ingested {stored_document.filename}: {len(stored_chunks)} chunks")
        except ValueError as exc:
            print(f"Skipped {file_path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local documents into the Phase 1 SQLite vector store.")
    parser.add_argument("path", type=Path, help="File or directory to ingest.")
    parser.add_argument("--db", type=Path, default=Path("data/processed/ragops.db"))
    args = parser.parse_args()

    ingest_path(args.path, SQLiteVectorStore(args.db))


if __name__ == "__main__":
    main()

