import io

from fastapi.testclient import TestClient

from app.main import app
from app.routers import image as image_router
from app.services.gpu_queue import GpuQueueFullError
from app.services.misinfo_runner import MisinfoInferenceError
from app.services.scam_runner import ScamInferenceError, ScamResult
from app.services.spai_runner import SpaiInferenceError, SpaiResult

client = TestClient(app)


def test_process_image_returns_score(monkeypatch):
    monkeypatch.setattr(
        image_router, "run_spai_inference", lambda data, filename: SpaiResult(0.87, None)
    )
    monkeypatch.setattr(
        image_router, "run_scam_inference_image", lambda data, filename: ScamResult(None, [], [])
    )

    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ai_detection": {"model": "spai", "score": 0.87, "evidence_image": None},
        "scam_detection": None,
        "misinformation_detection": None,
    }


def test_process_image_returns_evidence_image_when_present(monkeypatch):
    monkeypatch.setattr(
        image_router, "run_spai_inference", lambda data, filename: SpaiResult(0.87, "base64data")
    )
    monkeypatch.setattr(
        image_router, "run_scam_inference_image", lambda data, filename: ScamResult(None, [], [])
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
        image_router, "run_scam_inference_image", lambda data, filename: ScamResult(None, [], [])
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
        lambda data, filename: ScamResult(0.82, [{"sentence": "계좌번호를 알려주세요", "score": 0.95}], ["계좌번호를 알려주세요"]),
    )
    monkeypatch.setattr(
        image_router,
        "run_misinfo_inference",
        lambda sentences: type("R", (), {"model": "qwen3.5:4b", "wiki_snapshot": "2026-09-01", "claims": []})(),
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
        image_router, "run_scam_inference_image", lambda data, filename: ScamResult(None, [], [])
    )

    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
    )

    assert response.json()["scam_detection"] is None


def test_process_image_returns_502_when_scam_inference_fails(monkeypatch):
    # 텍스트가 없어서가 아니라 사기감지 파이프라인 자체가 죽은 경우, null로 조용히
    # 감추면 "사기 아님"으로 오인될 위험이 있어(2026-09-21) 전체 요청을 실패시킨다.
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

    assert response.status_code == 502


def test_process_image_returns_503_when_gpu_queue_full(monkeypatch):
    def raise_full(data, filename):
        raise GpuQueueFullError("GPU 자원 큐가 가득 찼습니다")

    monkeypatch.setattr(image_router, "run_spai_inference", raise_full)
    monkeypatch.setattr(
        image_router, "run_scam_inference_image", lambda data, filename: ScamResult(None, [], [])
    )

    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
    )

    assert response.status_code == 503


def test_process_image_includes_misinformation_detection_when_refuted(monkeypatch):
    monkeypatch.setattr(image_router, "run_spai_inference", lambda data, filename: SpaiResult(0.1, None))
    monkeypatch.setattr(
        image_router,
        "run_scam_inference_image",
        lambda data, filename: ScamResult(None, [], ["선풍기를 틀고 자면 사망한다."]),
    )
    monkeypatch.setattr(
        image_router,
        "run_misinfo_inference",
        lambda sentences: type(
            "R", (), {"model": "qwen3.5:4b", "wiki_snapshot": "2026-09-01",
                      "claims": [{"sentence": sentences[0], "reason": "미신이다.", "evidence": []}]}
        )(),
    )

    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["misinformation_detection"] == {
        "model": "qwen3.5:4b",
        "wiki_snapshot": "2026-09-01",
        "claims": [{"sentence": "선풍기를 틀고 자면 사망한다.", "reason": "미신이다.", "evidence": []}],
    }


def test_process_image_omits_misinformation_detection_when_no_sentences(monkeypatch):
    monkeypatch.setattr(image_router, "run_spai_inference", lambda data, filename: SpaiResult(0.1, None))
    monkeypatch.setattr(
        image_router, "run_scam_inference_image", lambda data, filename: ScamResult(None, [], [])
    )
    called = []
    monkeypatch.setattr(
        image_router, "run_misinfo_inference", lambda sentences: called.append(sentences)
    )

    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
    )

    assert response.json()["misinformation_detection"] is None
    assert called == []  # 문장이 없으면 가짜정보 판정 자체를 호출하지 않는다


def test_process_image_returns_502_when_misinfo_inference_fails(monkeypatch):
    monkeypatch.setattr(image_router, "run_spai_inference", lambda data, filename: SpaiResult(0.1, None))
    monkeypatch.setattr(
        image_router,
        "run_scam_inference_image",
        lambda data, filename: ScamResult(None, [], ["아무 문장"]),
    )

    def raise_error(sentences):
        raise MisinfoInferenceError("boom")

    monkeypatch.setattr(image_router, "run_misinfo_inference", raise_error)

    response = client.post(
        "/process/image",
        files={"file": ("test.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
    )

    assert response.status_code == 502
