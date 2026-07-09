from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class DocumentChunk(BaseModel):
    chunk_id: str
    chunk_index: int
    text: str
    token_count: int
    chunk_version: int
    embedding_model: str
    embedding_model_version: str


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    sha256: str
    document_version: int
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class UploadResponse(BaseModel):
    document: DocumentResponse
    chunks: list[DocumentChunk]
    message: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=4, ge=1, le=10)


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    score: float
    text: str


class AskResponse(BaseModel):
    trace_id: str
    answer: str
    citations: list[Citation]
    model: str
    confidence: float

