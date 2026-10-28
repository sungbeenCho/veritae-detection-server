import io

from fastapi.testclient import TestClient

from app import main
from app.main import app

client = TestClient(app)


def test_health_is_open_even_when_api_key_is_set(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "secret")

    assert client.get("/health").status_code == 200


def test_analysis_without_key_is_rejected_before_any_work(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "secret")

    response = client.post("/process/image", files={"file": ("a.jpg", io.BytesIO(b"x"), "image/jpeg")})

    assert response.status_code == 401


def test_analysis_with_wrong_key_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "secret")

    response = client.post(
        "/process/image",
        files={"file": ("a.jpg", io.BytesIO(b"x"), "image/jpeg")},
        headers={"X-Detection-Key": "wrong"},
    )

    assert response.status_code == 401


def test_analysis_with_right_key_reaches_the_router(monkeypatch):
    # 키가 맞으면 라우터까지 간다 - 여기서는 지원하지 않는 형식이라 라우터가 415를 돌려주는 것으로 확인한다.
    monkeypatch.setattr(main, "API_KEY", "secret")

    response = client.post(
        "/process/image",
        files={"file": ("a.txt", io.BytesIO(b"x"), "text/plain")},
        headers={"X-Detection-Key": "secret"},
    )

    assert response.status_code == 415


def test_no_key_configured_means_no_check(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "")

    response = client.post("/process/image", files={"file": ("a.txt", io.BytesIO(b"x"), "text/plain")})

    assert response.status_code == 415
