from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_rejects_unsupported_file() -> None:
    client = TestClient(app)
    response = client.post(
        "/documents/upload",
        files={"file": ("image.png", b"not supported", "image/png")},
    )

    assert response.status_code == 400


def test_ask_creates_trace_and_feedback() -> None:
    client = TestClient(app)
    upload_response = client.post(
        "/documents/upload",
        files={
            "file": (
                "phase-two-handbook.md",
                b"Runbooks explain incident response ownership and escalation steps.",
                "text/markdown",
            )
        },
    )
    assert upload_response.status_code == 200

    ask_response = client.post(
        "/ask",
        json={"question": "What explains incident response ownership?", "top_k": 2, "prompt_version": "v1"},
    )
    assert ask_response.status_code == 200
    ask_payload = ask_response.json()
    assert ask_payload["trace_id"]
    assert ask_payload["citations"]
    assert ask_payload["model"] == "local-extractive-rag-v1"

    trace_response = client.get(f'/traces/{ask_payload["trace_id"]}')
    assert trace_response.status_code == 200
    assert trace_response.json()["question"] == "What explains incident response ownership?"

    feedback_response = client.post(
        f'/traces/{ask_payload["trace_id"]}/feedback',
        json={"feedback": "up", "comment": "Grounded answer."},
    )
    assert feedback_response.status_code == 200
    assert feedback_response.json()["feedback_score"] == 1


def test_metrics_endpoint_exposes_prometheus_metrics() -> None:
    client = TestClient(app)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "ragops_requests_total" in response.text
