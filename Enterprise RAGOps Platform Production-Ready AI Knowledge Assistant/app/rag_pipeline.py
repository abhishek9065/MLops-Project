from __future__ import annotations

import uuid

from app.schemas import AskResponse, Citation
from app.vector_store import SQLiteVectorStore


class RagPipeline:
    def __init__(self, vector_store: SQLiteVectorStore) -> None:
        self.vector_store = vector_store

    def retrieve_only_answer(self, question: str, top_k: int = 4) -> AskResponse:
        chunks = self.vector_store.search(question, top_k=top_k)
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
        confidence = round(max((chunk.score for chunk in chunks), default=0.0), 4)
        answer = (
            "Phase 1 retrieval is working. LLM answer generation starts in Phase 2. "
            "Use the citations to inspect the retrieved document chunks."
        )
        return AskResponse(
            trace_id=str(uuid.uuid4()),
            answer=answer,
            citations=citations,
            model="retrieval-only-phase-1",
            confidence=confidence,
        )

