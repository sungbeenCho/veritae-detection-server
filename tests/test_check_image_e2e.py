"""실행 중인 서버에 이미지를 보내 가짜정보 탐지가 실제로 나오는지 확인하는 스크립트(scripts/check_image_e2e.py)의 테스트.
2026-10-13: 회귀 테스트는 통과하는데 서버로 돌리면 탐지가 안 됐다 - 서버를 통한 끝에서 끝까지 확인이 없었다."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from check_image_e2e import build_multipart, evaluate, run

SAMPLE_IMAGE = Path(__file__).parent / "samples" / "misinfo_test_image.png"

DETECTED = {
    "ai_detection": {"model": "spai", "score": 0.1, "evidence_image": None},
    "scam_detection": None,
    "misinformation_detection": {
        "model": "qwen3.5:4b",
        "wiki_snapshot": "2026-09-01",
        "claims": [
            {
                "sentence": "선풍기를 틀고 자면 사망한다는 이야기를 들었어요.",
                "reason": "'한국의 미신' 문서는 잘못된 정보라고 한다.",
                "evidence": [{"title": "한국의 미신", "text": "잘못된 정보를 믿는다.", "url": "https://ko.wikipedia.org/wiki/한국의_미신"}],
            }
        ],
    },
}
NOT_DETECTED = {**DETECTED, "misinformation_detection": {"model": "qwen3.5:4b", "wiki_snapshot": "2026-09-01", "claims": []}}


def test_sample_image_is_committed():
    assert SAMPLE_IMAGE.is_file()


def test_build_multipart_contains_filename_content_type_and_bytes():
    body, content_type = build_multipart("file", "a.png", "image/png", b"\x89PNGdata")

    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=")[1].encode()
    assert boundary in body
    assert b'name="file"; filename="a.png"' in body
    assert b"Content-Type: image/png" in body
    assert b"\x89PNGdata" in body


def test_evaluate_passes_when_expected_sentence_is_detected():
    ok, lines = evaluate(DETECTED, expect="선풍기")

    assert ok is True
    assert any("한국의 미신" in line for line in lines)


def test_evaluate_fails_when_nothing_is_detected():
    ok, lines = evaluate(NOT_DETECTED, expect="선풍기")

    assert ok is False
    assert any("탐지되지 않았" in line for line in lines)


def test_evaluate_fails_when_misinformation_field_is_missing():
    ok, _ = evaluate({"ai_detection": {}, "scam_detection": None, "misinformation_detection": None}, expect="선풍기")

    assert ok is False


def test_evaluate_fails_when_a_different_sentence_is_detected():
    ok, _ = evaluate(DETECTED, expect="에펠탑")

    assert ok is False


@pytest.fixture
def fake_server():
    """/process/image에 정해 둔 JSON을 돌려주는 진짜 HTTP 서버."""
    state = {"status": 200, "body": DETECTED, "received": b""}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["received"] = self.rfile.read(int(self.headers["Content-Length"]))
            state["path"] = self.path
            state["key"] = self.headers.get("X-Detection-Key")
            data = json.dumps(state["body"]).encode()
            self.send_response(state["status"])
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()


def test_run_sends_image_to_process_image_and_reports_success(fake_server):
    ok, lines = run(fake_server["url"], SAMPLE_IMAGE, expect="선풍기", timeout=10)

    assert ok is True
    assert fake_server["path"] == "/process/image"
    assert b"\x89PNG" in fake_server["received"]
    assert any("걸린 시간" in line for line in lines)


def test_run_reports_failure_when_server_returns_error_status(fake_server):
    fake_server["status"] = 502
    fake_server["body"] = {"detail": "가짜정보 판정 처리 중 오류가 발생했습니다."}

    ok, lines = run(fake_server["url"], SAMPLE_IMAGE, expect="선풍기", timeout=10)

    assert ok is False
    assert any("502" in line for line in lines)


def test_run_reports_failure_when_server_is_unreachable():
    ok, lines = run("http://127.0.0.1:9", SAMPLE_IMAGE, expect="선풍기", timeout=2)

    assert ok is False
    assert any("연결" in line for line in lines)


def test_run_sends_detection_key_header_when_env_is_set(fake_server, monkeypatch):
    # 2026-10-28부터 데스크탑 서버는 DETECTION_API_KEY가 맞지 않으면 401을 돌려준다 - 확인 스크립트도 같은 키를 보내야 한다.
    monkeypatch.setenv("DETECTION_API_KEY", "secret-key")

    ok, _ = run(fake_server["url"], SAMPLE_IMAGE, expect="선풍기", timeout=10)

    assert ok is True
    assert fake_server["key"] == "secret-key"


def test_run_omits_detection_key_header_when_env_is_empty(fake_server, monkeypatch):
    monkeypatch.delenv("DETECTION_API_KEY", raising=False)

    run(fake_server["url"], SAMPLE_IMAGE, expect="선풍기", timeout=10)

    assert fake_server["key"] is None
