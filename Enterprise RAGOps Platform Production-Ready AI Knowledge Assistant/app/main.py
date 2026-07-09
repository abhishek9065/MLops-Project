from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.document_loader import chunk_text, load_document, validate_supported_file
from app.rag_pipeline import RagPipeline
from app.schemas import AskRequest, AskResponse, DocumentChunk, DocumentResponse, HealthResponse, UploadResponse
from app.vector_store import SQLiteVectorStore, StoredDocument


APP_VERSION = "0.1.0-phase-1"
RAW_DATA_DIR = Path("data/raw")

app = FastAPI(
    title="Enterprise RAGOps Platform",
    description="Production-style RAG and LLMOps learning platform.",
    version=APP_VERSION,
)
vector_store = SQLiteVectorStore()
rag_pipeline = RagPipeline(vector_store)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="enterprise-ragops-platform", version=APP_VERSION)


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    try:
        validate_supported_file(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_filename = Path(file.filename or "uploaded-document").name
    destination = RAW_DATA_DIR / safe_filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        loaded_document = load_document(destination, content_type=file.content_type or "application/octet-stream")
        chunks = chunk_text(loaded_document.text)
        if not chunks:
            raise ValueError("Document did not produce any chunks.")
        stored_document, stored_chunks = vector_store.upsert_document(loaded_document, chunks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadResponse(
        document=to_document_response(stored_document),
        chunks=[
            DocumentChunk(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                token_count=chunk.token_count,
                chunk_version=chunk.chunk_version,
                embedding_model=chunk.embedding_model,
                embedding_model_version=chunk.embedding_model_version,
            )
            for chunk in stored_chunks
        ],
        message="Document uploaded, parsed, chunked, embedded, and stored locally.",
    )


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents() -> list[DocumentResponse]:
    return [to_document_response(document) for document in vector_store.list_documents()]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return rag_pipeline.retrieve_only_answer(request.question, top_k=request.top_k)


def to_document_response(document: StoredDocument) -> DocumentResponse:
    return DocumentResponse(
        document_id=document.document_id,
        filename=document.filename,
        content_type=document.content_type,
        sha256=document.sha256,
        document_version=document.document_version,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )

