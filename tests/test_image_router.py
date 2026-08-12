import io

from fastapi.testclient import TestClient

from app.main import app
from app.routers import image as image_router
from app.services.spai_runner import SpaiInferenceError

client = TestClient(app)


def test_process_image_returns_score(monkeypatch):
    monkeypatch.setattr(image_router, "run_spai_inference", lambda data, filename: 0.87)

    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {"ai_detection": {"model": "spai", "score": 0.87}}


def test_process_image_rejects_unsupported_content_type():
    response = client.post(
        "/process/image",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
    )

    assert response.status_code == 415


def test_process_image_rejects_empty_file():
    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b""), "image/jpeg")},
    )

    assert response.status_code == 400


def test_process_image_returns_502_on_spai_failure(monkeypatch):
    def raise_error(data, filename):
        raise SpaiInferenceError("boom")

    monkeypatch.setattr(image_router, "run_spai_inference", raise_error)

    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
    )

    assert response.status_code == 502
