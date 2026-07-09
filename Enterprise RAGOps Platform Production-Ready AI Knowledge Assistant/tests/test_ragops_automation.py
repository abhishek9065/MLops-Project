from pathlib import Path

from app.vector_store import SQLiteVectorStore
from scripts.delete_stale_chunks import delete_stale_documents
from scripts.reindex_changed_docs import reindex_changed_docs
from scripts.test_retrieval_quality import run_retrieval_quality_test
from scripts.run_evaluation import DEFAULT_DATASET, DEFAULT_SOURCE_DIR


def test_reindex_changed_docs_skips_unchanged_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    document_path = source_dir / "runbook.md"
    document_path.write_text("Every deployment needs a health check.", encoding="utf-8")
    store = SQLiteVectorStore(tmp_path / "ragops.db")

    first = reindex_changed_docs(source_dir, store)
    second = reindex_changed_docs(source_dir, store)

    assert first["indexed"] == 1
    assert second["skipped"] == 1


def test_delete_stale_documents_removes_missing_source_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    document_path = source_dir / "runbook.md"
    document_path.write_text("Every deployment needs a rollback plan.", encoding="utf-8")
    store = SQLiteVectorStore(tmp_path / "ragops.db")
    reindex_changed_docs(source_dir, store)

    document_path.unlink()
    stats = delete_stale_documents(source_dir, store)

    assert stats["deleted_missing_docs"] == 1
    assert store.list_documents() == []


def test_retrieval_quality_gate_passes() -> None:
    report = run_retrieval_quality_test(DEFAULT_DATASET, DEFAULT_SOURCE_DIR, threshold=0.75, top_k=4)

    assert report["passed"] is True
    assert report["score"] >= 0.75
