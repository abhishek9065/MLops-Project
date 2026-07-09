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

