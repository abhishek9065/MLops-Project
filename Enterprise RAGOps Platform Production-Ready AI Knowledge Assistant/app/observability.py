from __future__ import annotations

import logging
import os

from app.tracing import TraceRecord


def export_trace(record: TraceRecord) -> None:
    """Optional hook for Langfuse/OpenTelemetry without forcing those deps in Phase 4."""
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        export_to_opentelemetry(record)
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        logging.getLogger("ragops.observability").info(
            "langfuse_config_detected",
            extra={
                "trace_id": record.trace_id,
                "extra_payload": {
                    "note": "Install and wire the Langfuse SDK in a later phase for remote trace export."
                },
            },
        )


def export_to_opentelemetry(record: TraceRecord) -> None:
    try:
        from opentelemetry import trace
    except ImportError:
        logging.getLogger("ragops.observability").warning(
            "otel_endpoint_configured_without_package",
            extra={
                "trace_id": record.trace_id,
                "extra_payload": {"package": "opentelemetry-api"},
            },
        )
        return

    tracer = trace.get_tracer("enterprise-ragops-platform")
    with tracer.start_as_current_span("rag.ask") as span:
        span.set_attribute("rag.trace_id", record.trace_id)
        span.set_attribute("rag.prompt_version", record.prompt_version)
        span.set_attribute("rag.llm_model", record.llm_model)
        span.set_attribute("rag.embedding_model", record.embedding_model)
        span.set_attribute("rag.total_latency_ms", record.total_latency_ms)
        span.set_attribute("rag.estimated_cost_usd", record.estimated_cost_usd)
