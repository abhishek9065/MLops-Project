from __future__ import annotations

import argparse
from pathlib import Path

from app.document_loader import SUPPORTED_EXTENSIONS
from app.vector_store import SQLiteVectorStore


def delete_stale_documents(
    source_dir: Path,
    store: SQLiteVectorStore,
    prune_inactive_versions: bool = True,
) -> dict[str, int]:
    active_filenames = {
        path.name
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    deleted_missing_docs = store.delete_documents_not_in(active_filenames)
    deleted_inactive_versions = store.delete_inactive_document_versions() if prune_inactive_versions else 0
    return {
        "deleted_missing_docs": deleted_missing_docs,
        "deleted_inactive_versions": deleted_inactive_versions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete indexed documents that no longer exist and prune old versions.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--db", type=Path, default=Path("data/processed/ragops.db"))
    parser.add_argument("--keep-inactive-versions", action="store_true")
    args = parser.parse_args()

    stats = delete_stale_documents(
        args.source_dir,
        SQLiteVectorStore(args.db),
        prune_inactive_versions=not args.keep_inactive_versions,
    )
    print(
        "Deleted missing docs: "
        f"{stats['deleted_missing_docs']} | Deleted inactive versions: {stats['deleted_inactive_versions']}"
    )


if __name__ == "__main__":
    main()
