import io

from fastapi.testclient import TestClient

from app.main import app
from app.routers import audio as audio_router
from app.services.antideepfake_runner import AntiDeepfakeInferenceError, AntiDeepfakeResult
from app.services.scam_runner import ScamInferenceError, ScamResult

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
    monkeypatch.setattr(
        audio_router, "run_scam_inference_audio", lambda data, filename: ScamResult(None, [])
    )

    response = client.post(
        "/process/audio",
        files={"file": ("test.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scam_detection"] is None
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
    monkeypatch.setattr(
        audio_router, "run_scam_inference_audio", lambda data, filename: ScamResult(None, [])
    )

    response = client.post(
        "/process/audio",
        files={"file": ("test.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )

    assert response.status_code == 502


def test_process_audio_returns_scam_detection_when_present(monkeypatch):
    monkeypatch.setattr(
        audio_router,
        "run_antideepfake_inference",
        lambda data, filename: AntiDeepfakeResult(score=0.87, evidence=[]),
    )
    monkeypatch.setattr(
        audio_router,
        "run_scam_inference_audio",
        lambda data, filename: ScamResult(0.82, [{"sentence": "계좌번호를 알려주세요", "score": 0.95}]),
    )

    response = client.post(
        "/process/audio",
        files={"file": ("test.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["scam_detection"] == {
        "model": "lilju",
        "score": 0.82,
        "evidence": [{"sentence": "계좌번호를 알려주세요", "score": 0.95}],
    }


def test_process_audio_omits_scam_detection_when_no_text(monkeypatch):
    monkeypatch.setattr(
        audio_router,
        "run_antideepfake_inference",
        lambda data, filename: AntiDeepfakeResult(score=0.87, evidence=[]),
    )
    monkeypatch.setattr(
        audio_router, "run_scam_inference_audio", lambda data, filename: ScamResult(None, [])
    )

    response = client.post(
        "/process/audio",
        files={"file": ("test.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )

    assert response.json()["scam_detection"] is None


def test_process_audio_ignores_scam_inference_failure(monkeypatch):
    monkeypatch.setattr(
        audio_router,
        "run_antideepfake_inference",
        lambda data, filename: AntiDeepfakeResult(score=0.87, evidence=[]),
    )

    def raise_scam_error(data, filename):
        raise ScamInferenceError("boom")

    monkeypatch.setattr(audio_router, "run_scam_inference_audio", raise_scam_error)

    response = client.post(
        "/process/audio",
        files={"file": ("test.wav", io.BytesIO(b"fake-audio-bytes"), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["scam_detection"] is None
