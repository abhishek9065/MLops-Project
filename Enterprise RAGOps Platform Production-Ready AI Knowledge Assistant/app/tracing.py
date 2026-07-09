from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.schemas import Citation, TokenUsage, TraceResponse


@dataclass(frozen=True)
class TraceEvent:
    trace_id: str
    event_name: str
    payload: dict
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        if hasattr(record, "trace_id"):
            payload["trace_id"] = record.trace_id
        if hasattr(record, "extra_payload"):
            payload["payload"] = record.extra_payload
        return json.dumps(payload)


def configure_logging() -> None:
    root_logger = logging.getLogger()
    if any(isinstance(handler.formatter, JsonFormatter) for handler in root_logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class TraceRecord:
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
    error: str | None = None


class SQLiteTraceStore:
    def __init__(self, db_path: Path | str = "data/processed/ragops.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    llm_model TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    token_usage_json TEXT NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    retrieval_latency_ms REAL NOT NULL,
                    llm_latency_ms REAL NOT NULL,
                    total_latency_ms REAL NOT NULL,
                    feedback_score INTEGER,
                    feedback_comment TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def save_trace(self, record: TraceRecord) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO traces (
                    trace_id, question, answer, citations_json, prompt_version,
                    prompt, llm_model, embedding_model, token_usage_json,
                    estimated_cost_usd, retrieval_latency_ms, llm_latency_ms,
                    total_latency_ms, error, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.trace_id,
                    record.question,
                    record.answer,
                    json.dumps([citation.model_dump() for citation in record.citations]),
                    record.prompt_version,
                    record.prompt,
                    record.llm_model,
                    record.embedding_model,
                    record.token_usage.model_dump_json(),
                    record.estimated_cost_usd,
                    record.retrieval_latency_ms,
                    record.llm_latency_ms,
                    record.total_latency_ms,
                    record.error,
                    now,
                    now,
                ),
            )
        logging.getLogger("ragops.trace").info(
            "saved_trace",
            extra={
                "trace_id": record.trace_id,
                "extra_payload": {
                    "model": record.llm_model,
                    "prompt_version": record.prompt_version,
                    "estimated_cost_usd": record.estimated_cost_usd,
                    "total_latency_ms": record.total_latency_ms,
                    "citation_count": len(record.citations),
                },
            },
        )
        from app.observability import export_trace

        export_trace(record)

    def get_trace(self, trace_id: str) -> TraceResponse | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
        if row is None:
            return None
        return row_to_trace(row)

    def record_feedback(self, trace_id: str, feedback_score: int, comment: str | None = None) -> bool:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE traces
                SET feedback_score = ?, feedback_comment = ?, updated_at = ?
                WHERE trace_id = ?
                """,
                (feedback_score, comment, now, trace_id),
            )
        if cursor.rowcount:
            from app.metrics import record_feedback_metric

            record_feedback_metric(feedback_score)
            logging.getLogger("ragops.feedback").info(
                "recorded_feedback",
                extra={"trace_id": trace_id, "extra_payload": {"feedback_score": feedback_score}},
            )
        return cursor.rowcount > 0


def row_to_trace(row: sqlite3.Row) -> TraceResponse:
    token_payload = json.loads(row["token_usage_json"])
    return TraceResponse(
        trace_id=row["trace_id"],
        question=row["question"],
        answer=row["answer"],
        citations=[Citation(**payload) for payload in json.loads(row["citations_json"])],
        prompt_version=row["prompt_version"],
        prompt=row["prompt"],
        llm_model=row["llm_model"],
        embedding_model=row["embedding_model"],
        token_usage=TokenUsage(**token_payload),
        estimated_cost_usd=row["estimated_cost_usd"],
        retrieval_latency_ms=row["retrieval_latency_ms"],
        llm_latency_ms=row["llm_latency_ms"],
        total_latency_ms=row["total_latency_ms"],
        feedback_score=row["feedback_score"],
        feedback_comment=row["feedback_comment"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
