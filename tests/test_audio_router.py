import io

from fastapi.testclient import TestClient

from app.main import app
from app.routers import audio as audio_router
from app.services.antideepfake_runner import AntiDeepfakeInferenceError, AntiDeepfakeResult

client = TestClient(app)


def test_process_audio_returns_score_and_evidence(monkeypatch):
    monkeypatch.setattr(
        audio_router,
        "run_antideepfake_inference",
        lambda data, filename: AntiDeepfakeResult(
            score=0.87,
            evidence=[
                {
                    "title": "시간 구간 이상 패턴",
                    "description": "0.5초~1.2초 구간에서 합성 흔적이 감지됨",
                    "tags": ["temporal"],
                    "start_sec": 0.5,
                    "end_sec": 1.2,
                }
            ],
        ),
    )

    response = client.post(
        "/process/audio",
        files={"file": ("test.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ai_detection"]["model"] == "antideepfake"
    assert body["ai_detection"]["score"] == 0.87
    assert body["ai_detection"]["evidence"][0]["title"] == "시간 구간 이상 패턴"
    assert body["ai_detection"]["evidence"][0]["start_sec"] == 0.5


def test_process_audio_rejects_unsupported_content_type():
    response = client.post(
        "/process/audio",
        files={"file": ("test.txt", io.BytesIO(b"not audio"), "text/plain")},
    )

    assert response.status_code == 415


def test_process_audio_rejects_empty_file():
    response = client.post(
        "/process/audio",
        files={"file": ("test.wav", io.BytesIO(b""), "audio/wav")},
    )

    assert response.status_code == 400


def test_process_audio_returns_502_on_inference_failure(monkeypatch):
    def raise_error(data, filename):
        raise AntiDeepfakeInferenceError("boom")

    monkeypatch.setattr(audio_router, "run_antideepfake_inference", raise_error)

    response = client.post(
        "/process/audio",
        files={"file": ("test.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )

    assert response.status_code == 502
