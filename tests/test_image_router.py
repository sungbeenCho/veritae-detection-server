import io

from fastapi.testclient import TestClient

from app.main import app
from app.routers import image as image_router
from app.services.scam_runner import ScamInferenceError, ScamResult
from app.services.spai_runner import SpaiInferenceError, SpaiResult

client = TestClient(app)


def test_process_image_returns_score(monkeypatch):
    monkeypatch.setattr(
        image_router, "run_spai_inference", lambda data, filename: SpaiResult(0.87, None)
    )
    monkeypatch.setattr(
        image_router, "run_scam_inference_image", lambda data, filename: ScamResult(None, [])
    )

    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ai_detection": {"model": "spai", "score": 0.87, "evidence_image": None},
        "scam_detection": None,
    }


def test_process_image_returns_evidence_image_when_present(monkeypatch):
    monkeypatch.setattr(
        image_router, "run_spai_inference", lambda data, filename: SpaiResult(0.87, "base64data")
    )
    monkeypatch.setattr(
        image_router, "run_scam_inference_image", lambda data, filename: ScamResult(None, [])
    )

    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["ai_detection"]["evidence_image"] == "base64data"


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
    monkeypatch.setattr(
        image_router, "run_scam_inference_image", lambda data, filename: ScamResult(None, [])
    )

    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
    )

    assert response.status_code == 502


def test_process_image_returns_scam_detection_when_present(monkeypatch):
    monkeypatch.setattr(
        image_router, "run_spai_inference", lambda data, filename: SpaiResult(0.87, None)
    )
    monkeypatch.setattr(
        image_router,
        "run_scam_inference_image",
        lambda data, filename: ScamResult(0.82, [{"sentence": "계좌번호를 알려주세요", "score": 0.95}]),
    )

    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["scam_detection"] == {
        "model": "lilju",
        "score": 0.82,
        "evidence": [{"sentence": "계좌번호를 알려주세요", "score": 0.95}],
    }


def test_process_image_omits_scam_detection_when_no_text(monkeypatch):
    monkeypatch.setattr(
        image_router, "run_spai_inference", lambda data, filename: SpaiResult(0.87, None)
    )
    monkeypatch.setattr(
        image_router, "run_scam_inference_image", lambda data, filename: ScamResult(None, [])
    )

    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
    )

    assert response.json()["scam_detection"] is None


def test_process_image_ignores_scam_inference_failure(monkeypatch):
    monkeypatch.setattr(
        image_router, "run_spai_inference", lambda data, filename: SpaiResult(0.87, None)
    )

    def raise_scam_error(data, filename):
        raise ScamInferenceError("boom")

    monkeypatch.setattr(image_router, "run_scam_inference_image", raise_scam_error)

    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["scam_detection"] is None
