from __future__ import annotations

import argparse
from pathlib import Path

from app.document_loader import chunk_text, load_document
from app.vector_store import SQLiteVectorStore
from scripts.delete_stale_chunks import delete_stale_documents


def reindex_changed_docs(source_dir: Path, store: SQLiteVectorStore) -> dict[str, int]:
    stats = {"indexed": 0, "skipped": 0, "failed": 0}
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        try:
            document = load_document(path)
            existing = store.get_latest_document_by_filename(document.filename)
            if existing and existing.sha256 == document.sha256:
                stats["skipped"] += 1
                continue
            store.upsert_document(document, chunk_text(document.text))
            stats["indexed"] += 1
        except ValueError as exc:
            stats["failed"] += 1
            print(f"Skipped {path}: {exc}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Reindex only documents whose content hash changed.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--db", type=Path, default=Path("data/processed/ragops.db"))
    parser.add_argument("--delete-stale", action="store_true")
    args = parser.parse_args()

    store = SQLiteVectorStore(args.db)
    stats = reindex_changed_docs(args.source_dir, store)
    print(f"Indexed: {stats['indexed']} | Skipped unchanged: {stats['skipped']} | Failed: {stats['failed']}")
    if args.delete_stale:
        stale_stats = delete_stale_documents(args.source_dir, store)
        print(
            "Deleted missing docs: "
            f"{stale_stats['deleted_missing_docs']} | "
            f"Deleted inactive versions: {stale_stats['deleted_inactive_versions']}"
        )


if __name__ == "__main__":
    main()
