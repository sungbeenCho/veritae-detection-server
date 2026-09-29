import io

from fastapi.testclient import TestClient

from app.main import app
from app.routers import video as video_router
from app.services.dfdc_runner import DfdcInferenceError, DfdcResult, NoFaceDetectedError
from app.services.gpu_queue import GpuQueueFullError
from app.services.scam_runner import ScamInferenceError, ScamResult

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
    monkeypatch.setattr(
        video_router, "run_scam_inference_video", lambda data, filename: ScamResult(None, [], [])
    )

    response = client.post(
        "/process/video",
        files={"file": ("test.mp4", io.BytesIO(b"fake-video-bytes"), "video/mp4")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scam_detection"] is None
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
    monkeypatch.setattr(
        video_router, "run_scam_inference_video", lambda data, filename: ScamResult(None, [], [])
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
    monkeypatch.setattr(
        video_router, "run_scam_inference_video", lambda data, filename: ScamResult(None, [], [])
    )

    response = client.post(
        "/process/video",
        files={"file": ("test.mp4", io.BytesIO(b"fake-video-bytes"), "video/mp4")},
    )

    assert response.status_code == 502


def test_process_video_returns_200_with_error_code_when_no_face_detected_and_no_scam_text(monkeypatch):
    def raise_no_face(data, filename):
        raise NoFaceDetectedError("얼굴을 찾을 수 없습니다.")

    monkeypatch.setattr(video_router, "run_dfdc_inference", raise_no_face)
    monkeypatch.setattr(
        video_router, "run_scam_inference_video", lambda data, filename: ScamResult(None, [], [])
    )

    response = client.post(
        "/process/video",
        files={"file": ("test.mp4", io.BytesIO(b"fake-video-bytes"), "video/mp4")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ai_detection"] is None
    assert body["scam_detection"] is None
    assert body["error_code"] == "NO_FACE_DETECTED"


def test_process_video_keeps_scam_detection_when_no_face_detected(monkeypatch):
    # 얼굴이 없어 AI판독은 못 해도, 별개로 돌아가는 사기감지 결과는 버리지 않고
    # 살려서 보여준다(2026-09-21) - 이번 수정의 핵심 케이스.
    def raise_no_face(data, filename):
        raise NoFaceDetectedError("얼굴을 찾을 수 없습니다.")

    monkeypatch.setattr(video_router, "run_dfdc_inference", raise_no_face)
    monkeypatch.setattr(
        video_router,
        "run_scam_inference_video",
        lambda data, filename: ScamResult(0.82, [{"sentence": "계좌번호를 알려주세요", "score": 0.95}], ["계좌번호를 알려주세요"]),
    )

    response = client.post(
        "/process/video",
        files={"file": ("test.mp4", io.BytesIO(b"fake-video-bytes"), "video/mp4")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ai_detection"] is None
    assert body["error_code"] == "NO_FACE_DETECTED"
    assert body["scam_detection"] == {
        "model": "lilju",
        "score": 0.82,
        "evidence": [{"sentence": "계좌번호를 알려주세요", "score": 0.95}],
    }


def test_process_video_returns_scam_detection_when_present(monkeypatch):
    monkeypatch.setattr(
        video_router,
        "run_dfdc_inference",
        lambda data, filename: DfdcResult(score=0.91, evidence=[], evidence_image=None),
    )
    monkeypatch.setattr(
        video_router,
        "run_scam_inference_video",
        lambda data, filename: ScamResult(0.82, [{"sentence": "계좌번호를 알려주세요", "score": 0.95}], ["계좌번호를 알려주세요"]),
    )

    response = client.post(
        "/process/video",
        files={"file": ("test.mp4", io.BytesIO(b"fake-video-bytes"), "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["scam_detection"] == {
        "model": "lilju",
        "score": 0.82,
        "evidence": [{"sentence": "계좌번호를 알려주세요", "score": 0.95}],
    }


def test_process_video_omits_scam_detection_when_no_text(monkeypatch):
    monkeypatch.setattr(
        video_router,
        "run_dfdc_inference",
        lambda data, filename: DfdcResult(score=0.91, evidence=[], evidence_image=None),
    )
    monkeypatch.setattr(
        video_router, "run_scam_inference_video", lambda data, filename: ScamResult(None, [], [])
    )

    response = client.post(
        "/process/video",
        files={"file": ("test.mp4", io.BytesIO(b"fake-video-bytes"), "video/mp4")},
    )

    assert response.json()["scam_detection"] is None


def test_process_video_returns_502_when_scam_inference_fails(monkeypatch):
    # 텍스트가 없어서가 아니라 사기감지 파이프라인 자체가 죽은 경우, null로 조용히
    # 감추면 "사기 아님"으로 오인될 위험이 있어(2026-09-21) 전체 요청을 실패시킨다.
    monkeypatch.setattr(
        video_router,
        "run_dfdc_inference",
        lambda data, filename: DfdcResult(score=0.91, evidence=[], evidence_image=None),
    )

    def raise_scam_error(data, filename):
        raise ScamInferenceError("boom")

    monkeypatch.setattr(video_router, "run_scam_inference_video", raise_scam_error)

    response = client.post(
        "/process/video",
        files={"file": ("test.mp4", io.BytesIO(b"fake-video-bytes"), "video/mp4")},
    )

    assert response.status_code == 502


def test_process_video_returns_503_when_gpu_queue_full(monkeypatch):
    def raise_full(data, filename):
        raise GpuQueueFullError("GPU 자원 큐가 가득 찼습니다")

    monkeypatch.setattr(video_router, "run_dfdc_inference", raise_full)
    monkeypatch.setattr(
        video_router, "run_scam_inference_video", lambda data, filename: ScamResult(None, [], [])
    )

    response = client.post(
        "/process/video",
        files={"file": ("test.mp4", io.BytesIO(b"fake-video-bytes"), "video/mp4")},
    )

    assert response.status_code == 503
