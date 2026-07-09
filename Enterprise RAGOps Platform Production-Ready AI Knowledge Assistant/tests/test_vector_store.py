from app.document_loader import LoadedDocument, chunk_text
from app.vector_store import SQLiteVectorStore
from scripts.delete_stale_chunks import delete_stale_documents


def test_vector_store_upsert_and_search(tmp_path) -> None:
    store = SQLiteVectorStore(tmp_path / "ragops.db")
    document = LoadedDocument(
        filename="handbook.md",
        content_type="text/markdown",
        sha256="abc123",
        text="Incident response requires alerts, owners, runbooks, and postmortems.",
    )
    stored_document, stored_chunks = store.upsert_document(
        document,
        chunk_text(document.text, chunk_size=20, overlap=0),
    )

    results = store.search("What requires runbooks?", top_k=1)

    assert stored_document.document_version == 1
    assert len(stored_chunks) == 1
    assert results[0].filename == "handbook.md"
    assert results[0].score > 0


def test_updated_document_deactivates_old_version(tmp_path) -> None:
    store = SQLiteVectorStore(tmp_path / "ragops.db")
    first = LoadedDocument(
        filename="runbook.md",
        content_type="text/markdown",
        sha256="v1",
        text="Old rollback instructions mention blue green deployment.",
    )
    second = LoadedDocument(
        filename="runbook.md",
        content_type="text/markdown",
        sha256="v2",
        text="New rollback instructions require restoring the previous vector index snapshot.",
    )

    first_doc, _ = store.upsert_document(first, chunk_text(first.text, chunk_size=20, overlap=0))
    second_doc, _ = store.upsert_document(second, chunk_text(second.text, chunk_size=20, overlap=0))
    results = store.search("vector index snapshot", top_k=5)
    all_documents = store.list_all_documents()

    assert first_doc.document_version == 1
    assert second_doc.document_version == 2
    assert [document.active for document in all_documents] == [True, False]
    assert all("Old rollback" not in result.text for result in results)


def test_delete_stale_documents_prunes_missing_and_inactive_versions(tmp_path) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    active_path = source_dir / "active.md"
    active_path.write_text("Active document about alerts.", encoding="utf-8")

    store = SQLiteVectorStore(tmp_path / "ragops.db")
    old_doc = LoadedDocument("active.md", "text/markdown", "old", "Old active document.")
    new_doc = LoadedDocument("active.md", "text/markdown", "new", "New active document.")
    removed_doc = LoadedDocument("removed.md", "text/markdown", "removed", "Removed document.")

    store.upsert_document(old_doc, chunk_text(old_doc.text, chunk_size=20, overlap=0))
    store.upsert_document(new_doc, chunk_text(new_doc.text, chunk_size=20, overlap=0))
    store.upsert_document(removed_doc, chunk_text(removed_doc.text, chunk_size=20, overlap=0))

    stats = delete_stale_documents(source_dir, store)
    remaining = store.list_all_documents()

    assert stats == {"deleted_missing_docs": 1, "deleted_inactive_versions": 1}
    assert len(remaining) == 1
    assert remaining[0].filename == "active.md"
    assert remaining[0].active is True
