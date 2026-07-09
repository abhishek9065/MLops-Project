from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


REQUESTS_TOTAL = Counter(
    "ragops_requests_total",
    "Total HTTP requests handled by the API.",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY_SECONDS = Histogram(
    "ragops_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["method", "endpoint"],
)

ERRORS_TOTAL = Counter(
    "ragops_errors_total",
    "Total HTTP responses with status code >= 500.",
    ["method", "endpoint"],
)

RETRIEVAL_LATENCY_SECONDS = Histogram(
    "ragops_retrieval_latency_seconds",
    "Vector retrieval latency in seconds.",
    ["embedding_model"],
)

LLM_LATENCY_SECONDS = Histogram(
    "ragops_llm_latency_seconds",
    "LLM generation latency in seconds.",
    ["llm_model", "prompt_version"],
)

TOKEN_USAGE_TOTAL = Counter(
    "ragops_token_usage_total",
    "Total prompt and completion tokens used by RAG answers.",
    ["llm_model", "token_type"],
)

ESTIMATED_COST_USD_TOTAL = Counter(
    "ragops_estimated_cost_usd_total",
    "Total estimated LLM cost in USD.",
    ["llm_model"],
)

FEEDBACK_SCORE_TOTAL = Counter(
    "ragops_feedback_score_total",
    "Total feedback score where thumbs up is 1 and thumbs down is -1.",
    ["score"],
)

LAST_FEEDBACK_SCORE = Gauge(
    "ragops_last_feedback_score",
    "Most recent feedback score.",
)


def record_http_request(method: str, endpoint: str, status_code: int, latency_seconds: float) -> None:
    REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
    REQUEST_LATENCY_SECONDS.labels(method=method, endpoint=endpoint).observe(latency_seconds)
    if status_code >= 500:
        ERRORS_TOTAL.labels(method=method, endpoint=endpoint).inc()


def record_rag_metrics(
    *,
    embedding_model: str,
    llm_model: str,
    prompt_version: str,
    retrieval_latency_ms: float,
    llm_latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    estimated_cost_usd: float,
) -> None:
    RETRIEVAL_LATENCY_SECONDS.labels(embedding_model=embedding_model).observe(retrieval_latency_ms / 1000)
    LLM_LATENCY_SECONDS.labels(llm_model=llm_model, prompt_version=prompt_version).observe(llm_latency_ms / 1000)
    TOKEN_USAGE_TOTAL.labels(llm_model=llm_model, token_type="prompt").inc(prompt_tokens)
    TOKEN_USAGE_TOTAL.labels(llm_model=llm_model, token_type="completion").inc(completion_tokens)
    ESTIMATED_COST_USD_TOTAL.labels(llm_model=llm_model).inc(estimated_cost_usd)


def record_feedback_metric(feedback_score: int) -> None:
    FEEDBACK_SCORE_TOTAL.labels(score=str(feedback_score)).inc(abs(feedback_score))
    LAST_FEEDBACK_SCORE.set(feedback_score)


def prometheus_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
