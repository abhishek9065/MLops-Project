from __future__ import annotations

import shutil
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile

from app.document_loader import chunk_text, load_document, validate_supported_file
from app.metrics import prometheus_payload, record_http_request
from app.rag_pipeline import RagPipeline
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentChunk,
    DocumentResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    TraceResponse,
    UploadResponse,
)
from app.tracing import SQLiteTraceStore, configure_logging
from app.vector_store import SQLiteVectorStore, StoredDocument


APP_VERSION = "0.4.0-phase-8"
RAW_DATA_DIR = Path("data/raw")

configure_logging()
app = FastAPI(
    title="Enterprise RAGOps Platform",
    description="Production-style RAG and LLMOps learning platform.",
    version=APP_VERSION,
)
vector_store = SQLiteVectorStore()
trace_store = SQLiteTraceStore()
rag_pipeline = RagPipeline(vector_store, trace_store)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    started_at = time.perf_counter()
    response = await call_next(request)
    endpoint = request.scope.get("route").path if request.scope.get("route") else request.url.path
    record_http_request(
        method=request.method,
        endpoint=endpoint,
        status_code=response.status_code,
        latency_seconds=time.perf_counter() - started_at,
    )
    return response


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
    try:
        return rag_pipeline.answer(
            request.question,
            top_k=request.top_k,
            prompt_version=request.prompt_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/traces/{trace_id}", response_model=TraceResponse)
def get_trace(trace_id: str) -> TraceResponse:
    trace = trace_store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace


@app.post("/traces/{trace_id}/feedback", response_model=FeedbackResponse)
def record_feedback(trace_id: str, request: FeedbackRequest) -> FeedbackResponse:
    feedback_score = 1 if request.feedback == "up" else -1
    updated = trace_store.record_feedback(trace_id, feedback_score, request.comment)
    if not updated:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return FeedbackResponse(
        trace_id=trace_id,
        feedback_score=feedback_score,
        message="Feedback recorded.",
    )


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = prometheus_payload()
    return Response(content=payload, media_type=content_type)


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
