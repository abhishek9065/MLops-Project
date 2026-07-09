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
    prompt_version: str = Field(default="v1")


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
    prompt_version: str
    embedding_model: str
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    token_usage: "TokenUsage"
    estimated_cost_usd: float


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TraceResponse(BaseModel):
    trace_id: str
    question: str
    answer: str
    citations: list[Citation]
    prompt_version: str
    prompt: str
    llm_model: str
    embedding_model: str
    token_usage: TokenUsage
    estimated_cost_usd: float
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    feedback_score: int | None = None
    feedback_comment: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class FeedbackRequest(BaseModel):
    feedback: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    trace_id: str
    feedback_score: int
    message: str
