import io

from fastapi.testclient import TestClient

from app.main import app
from app.routers import video as video_router
from app.services.dfdc_runner import DfdcInferenceError, DfdcResult

client = TestClient(app)


def test_process_video_returns_score_evidence_and_evidence_image(monkeypatch):
    monkeypatch.setattr(
        video_router,
        "run_dfdc_inference",
        lambda data, filename: DfdcResult(
            score=0.91,
            evidence=[
                {
                    "title": "얼굴 조작 의심 구간",
                    "description": "3.0초~7.0초 구간에서 얼굴 합성 흔적이 감지됨",
                    "tags": ["temporal", "face-swap"],
                    "start_sec": 3.0,
                    "end_sec": 7.0,
                }
            ],
            evidence_image="base64pngdata",
        ),
    )

    response = client.post(
        "/process/video",
        files={"file": ("test.mp4", io.BytesIO(b"fake-video-bytes"), "video/mp4")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ai_detection"]["model"] == "dfdc"
    assert body["ai_detection"]["score"] == 0.91
    assert body["ai_detection"]["evidence_image"] == "base64pngdata"
    assert body["ai_detection"]["evidence"][0]["title"] == "얼굴 조작 의심 구간"


def test_process_video_with_null_evidence_image(monkeypatch):
    monkeypatch.setattr(
        video_router,
        "run_dfdc_inference",
        lambda data, filename: DfdcResult(score=0.1, evidence=[], evidence_image=None),
    )

    response = client.post(
        "/process/video",
        files={"file": ("test.mp4", io.BytesIO(b"fake-video-bytes"), "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["ai_detection"]["evidence_image"] is None


def test_process_video_rejects_unsupported_content_type():
    response = client.post(
        "/process/video",
        files={"file": ("test.txt", io.BytesIO(b"not a video"), "text/plain")},
    )

    assert response.status_code == 415


def test_process_video_rejects_empty_file():
    response = client.post(
        "/process/video",
        files={"file": ("test.mp4", io.BytesIO(b""), "video/mp4")},
    )

    assert response.status_code == 400


def test_process_video_returns_502_on_inference_failure(monkeypatch):
    def raise_error(data, filename):
        raise DfdcInferenceError("boom")

    monkeypatch.setattr(video_router, "run_dfdc_inference", raise_error)

    response = client.post(
        "/process/video",
        files={"file": ("test.mp4", io.BytesIO(b"fake-video-bytes"), "video/mp4")},
    )

    assert response.status_code == 502
