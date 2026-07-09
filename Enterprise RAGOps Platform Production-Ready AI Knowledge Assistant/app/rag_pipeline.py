from __future__ import annotations

import time
import uuid
from pathlib import Path

from app.llm_client import LLMClient
from app.metrics import record_rag_metrics
from app.schemas import AskResponse, Citation
from app.tracing import SQLiteTraceStore, TraceRecord
from app.vector_store import SQLiteVectorStore


class RagPipeline:
    def __init__(
        self,
        vector_store: SQLiteVectorStore,
        trace_store: SQLiteTraceStore,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.trace_store = trace_store
        self.llm_client = llm_client or LLMClient()

    def answer(self, question: str, top_k: int = 4, prompt_version: str = "v1") -> AskResponse:
        trace_id = str(uuid.uuid4())
        total_started_at = time.perf_counter()
        retrieval_started_at = time.perf_counter()
        chunks = self.vector_store.search(question, top_k=top_k)
        retrieval_latency_ms = elapsed_ms(retrieval_started_at)
        citations = [
            Citation(
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                score=round(chunk.score, 4),
                text=chunk.text,
            )
            for chunk in chunks
        ]
        prompt = render_prompt(prompt_version=prompt_version, question=question, citations=citations)
        llm_result = self.llm_client.generate(prompt=prompt, question=question, citations=citations)
        confidence = round(max((chunk.score for chunk in chunks), default=0.0), 4)
        total_latency_ms = elapsed_ms(total_started_at)

        response = AskResponse(
            trace_id=trace_id,
            answer=llm_result.answer,
            citations=citations,
            model=llm_result.model,
            confidence=confidence,
            prompt_version=prompt_version,
            embedding_model=self.vector_store.embedding_model.name,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_result.latency_ms,
            total_latency_ms=total_latency_ms,
            token_usage=llm_result.token_usage,
            estimated_cost_usd=llm_result.estimated_cost_usd,
        )
        self.trace_store.save_trace(
            TraceRecord(
                trace_id=trace_id,
                question=question,
                answer=llm_result.answer,
                citations=citations,
                prompt_version=prompt_version,
                prompt=prompt,
                llm_model=llm_result.model,
                embedding_model=self.vector_store.embedding_model.name,
                token_usage=llm_result.token_usage,
                estimated_cost_usd=llm_result.estimated_cost_usd,
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=llm_result.latency_ms,
                total_latency_ms=total_latency_ms,
            )
        )
        record_rag_metrics(
            embedding_model=self.vector_store.embedding_model.name,
            llm_model=llm_result.model,
            prompt_version=prompt_version,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_result.latency_ms,
            prompt_tokens=llm_result.token_usage.prompt_tokens,
            completion_tokens=llm_result.token_usage.completion_tokens,
            estimated_cost_usd=llm_result.estimated_cost_usd,
        )
        return response


def render_prompt(prompt_version: str, question: str, citations: list[Citation]) -> str:
    prompt_path = Path("prompts") / f"rag_prompt_{prompt_version}.txt"
    if not prompt_path.exists():
        raise ValueError(f"Prompt version '{prompt_version}' does not exist at {prompt_path}.")
    template = prompt_path.read_text(encoding="utf-8")
    context = format_context(citations)
    return template.replace("{{ question }}", question).replace("{{ context }}", context)


def format_context(citations: list[Citation]) -> str:
    if not citations:
        return "No retrieved context."
    return "\n\n".join(
        (
            f"Source: {citation.filename}, chunk {citation.chunk_index}, "
            f"chunk_id {citation.chunk_id}, retrieval_score {citation.score}\n"
            f"{citation.text}"
        )
        for citation in citations
    )


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
