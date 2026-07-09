from app.document_loader import LoadedDocument, chunk_text
from app.vector_store import SQLiteVectorStore


def test_vector_store_upsert_and_search(tmp_path) -> None:
    store = SQLiteVectorStore(tmp_path / "ragops.db")
    document = LoadedDocument(
        filename="handbook.md",
        content_type="text/markdown",
        sha256="abc123",
        text="Incident response requires alerts, owners, runbooks, and postmortems.",
    )
    stored_document, stored_chunks = store.upsert_document(document, chunk_text(document.text, chunk_size=20))

    results = store.search("What requires runbooks?", top_k=1)

    assert stored_document.document_version == 1
    assert len(stored_chunks) == 1
    assert results[0].filename == "handbook.md"
    assert results[0].score > 0

