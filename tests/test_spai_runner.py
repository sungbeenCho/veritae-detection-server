import base64
import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.spai_runner import _find_and_encode_overlay, _safe_filename


@pytest.fixture(autouse=True)
def _stub_gpu_queue(monkeypatch):
    # run_spai_inference가 이제 get_gpu_queue().acquire(...)를 거친다. 실제
    # get_gpu_queue()는 app.config.get_settings()를 호출하는데, 이 테스트 파일의 기존
    # 테스트들은 spai_runner.get_settings만 목킹하고 gpu_queue 쪽 설정은 세팅하지 않으므로
    # 그대로 두면 환경변수 미설정 RuntimeError로 깨진다. 큐 동작 자체를 검증하는
    # test_run_spai_inference_uses_gpu_queue는 이 기본값을 자기 안에서 다시 덮어쓴다.
    # nullcontext를 쓰는 이유: MagicMock을 컨텍스트 매니저로 그냥 쓰면 __exit__이 기본적으로
    # truthy를 반환해 with 블록 안에서 난 예외(TimeoutExpired 등)를 조용히 삼켜버린다.
    from contextlib import nullcontext

    class _NoopGpuQueue:
        def acquire(self, label):
            return nullcontext()

    monkeypatch.setattr("app.services.spai_runner.get_gpu_queue", lambda: _NoopGpuQueue())


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("photo.jpg", "photo.jpg"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\Windows\\System32\\evil.dll", "evil.dll"),
        ("/etc/passwd", "passwd"),
        ("C:\\Windows\\System32\\evil.dll", "evil.dll"),
        ("..", "upload"),
        ("", "upload"),
    ],
)
def test_safe_filename_strips_path_traversal(raw, expected):
    assert _safe_filename(raw) == expected


def test_find_and_encode_overlay_returns_base64_when_file_exists(tmp_path):
    overlay_dir = tmp_path / "images" / "0" / "patches_attn"
    overlay_dir.mkdir(parents=True)
    (overlay_dir / "attn_overlay_0.87.png").write_bytes(b"fake-png-bytes")

    result = _find_and_encode_overlay(tmp_path)

    assert result == base64.b64encode(b"fake-png-bytes").decode("ascii")


def test_find_and_encode_overlay_returns_none_when_file_missing(tmp_path):
    assert _find_and_encode_overlay(tmp_path) is None


def test_find_and_encode_overlay_returns_none_on_unreadable_file(tmp_path, monkeypatch):
    overlay_dir = tmp_path / "images" / "0" / "patches_attn"
    overlay_dir.mkdir(parents=True)
    (overlay_dir / "attn_overlay_0.87.png").write_bytes(b"fake-png-bytes")

    def raise_error(self, *args, **kwargs):
        raise OSError("disk exploded")

    monkeypatch.setattr("pathlib.Path.read_bytes", raise_error)

    assert _find_and_encode_overlay(tmp_path) is None


def _write_score_csv(output_dir: Path, score: str) -> None:
    with open(output_dir / "input.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["spai"])
        writer.writeheader()
        writer.writerow({"spai": score})


def _spai_settings(tmp_path) -> MagicMock:
    settings = MagicMock()
    settings.work_dir = tmp_path
    settings.spai_python = "python"
    settings.spai_cfg = "./configs/spai.yaml"
    settings.spai_model = "./weights/spai.pth"
    settings.spai_resize_to = 1024
    settings.spai_repo_dir = tmp_path
    settings.spai_timeout_seconds = 120
    return settings


@patch("app.services.spai_runner.subprocess.run")
@patch("app.services.spai_runner.get_settings")
def test_run_spai_inference_returns_score_and_evidence_image(mock_get_settings, mock_run, tmp_path):
    from app.services.spai_runner import run_spai_inference

    mock_get_settings.return_value = _spai_settings(tmp_path)

    def fake_run(command, **kwargs):
        output_dir = Path(command[command.index("--output") + 1])
        _write_score_csv(output_dir, "0.87")
        overlay_dir = output_dir / "images" / "0" / "patches_attn"
        overlay_dir.mkdir(parents=True)
        (overlay_dir / "attn_overlay_0.87.png").write_bytes(b"fake-png")
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    result = run_spai_inference(b"fake-image-bytes", "test.jpg")

    assert result.score == 0.87
    assert result.evidence_image == base64.b64encode(b"fake-png").decode("ascii")


@patch("app.services.spai_runner.subprocess.run")
@patch("app.services.spai_runner.get_settings")
def test_run_spai_inference_omits_evidence_image_below_score_threshold(mock_get_settings, mock_run, tmp_path):
    """음성(antideepfake_infer.py)/영상(dfdc_infer.py)과 동일하게 score < 0.5 면 히트맵
    파일이 실제로 존재하더라도 evidence_image를 포함하지 않는다(2026-09-12, 사용자 확인)."""
    from app.services.spai_runner import run_spai_inference

    mock_get_settings.return_value = _spai_settings(tmp_path)

    def fake_run(command, **kwargs):
        output_dir = Path(command[command.index("--output") + 1])
        _write_score_csv(output_dir, "0.49")
        overlay_dir = output_dir / "images" / "0" / "patches_attn"
        overlay_dir.mkdir(parents=True)
        (overlay_dir / "attn_overlay_0.49.png").write_bytes(b"fake-png")
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    result = run_spai_inference(b"fake-image-bytes", "test.jpg")

    assert result.score == 0.49
    assert result.evidence_image is None


@patch("app.services.spai_runner.subprocess.run")
@patch("app.services.spai_runner.get_settings")
def test_run_spai_inference_includes_evidence_image_at_score_threshold(mock_get_settings, mock_run, tmp_path):
    """경계값(0.5 자체)은 포함돼야 한다 - antideepfake_infer.py/dfdc_infer.py와 동일하게
    >= 비교."""
    from app.services.spai_runner import run_spai_inference

    mock_get_settings.return_value = _spai_settings(tmp_path)

    def fake_run(command, **kwargs):
        output_dir = Path(command[command.index("--output") + 1])
        _write_score_csv(output_dir, "0.5")
        overlay_dir = output_dir / "images" / "0" / "patches_attn"
        overlay_dir.mkdir(parents=True)
        (overlay_dir / "attn_overlay_0.5.png").write_bytes(b"fake-png")
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    result = run_spai_inference(b"fake-image-bytes", "test.jpg")

    assert result.evidence_image == base64.b64encode(b"fake-png").decode("ascii")


@patch("app.services.spai_runner.subprocess.run")
@patch("app.services.spai_runner.get_settings")
def test_run_spai_inference_evidence_image_none_when_export_missing(mock_get_settings, mock_run, tmp_path):
    from app.services.spai_runner import run_spai_inference

    mock_get_settings.return_value = _spai_settings(tmp_path)

    def fake_run(command, **kwargs):
        output_dir = Path(command[command.index("--output") + 1])
        _write_score_csv(output_dir, "0.02")
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    result = run_spai_inference(b"fake-image-bytes", "test.jpg")

    assert result.score == 0.02
    assert result.evidence_image is None


@patch("app.services.spai_runner.subprocess.run")
@patch("app.services.spai_runner.get_settings")
def test_run_spai_inference_passes_export_image_patches_opt(mock_get_settings, mock_run, tmp_path):
    from app.services.spai_runner import run_spai_inference

    mock_get_settings.return_value = _spai_settings(tmp_path)
    captured_command = {}

    def fake_run(command, **kwargs):
        captured_command["value"] = command
        output_dir = Path(command[command.index("--output") + 1])
        _write_score_csv(output_dir, "0.5")
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    run_spai_inference(b"fake-image-bytes", "test.jpg")

    command = captured_command["value"]
    opt_idx = command.index("--opt")
    assert command[opt_idx + 1] == "TEST.EXPORT_IMAGE_PATCHES"
    assert command[opt_idx + 2] == "True"


@patch("app.services.spai_runner.subprocess.run")
@patch("app.services.spai_runner.get_settings")
def test_run_spai_inference_uses_gpu_queue(mock_get_settings, mock_run, monkeypatch, tmp_path):
    calls = []

    class FakeQueue:
        def acquire(self, label):
            calls.append(label)
            from contextlib import contextmanager

            @contextmanager
            def cm():
                yield

            return cm()

    # gpu_queue_module.get_gpu_queue가 아니라 spai_runner가 직접 import해서 쓰는 이름을
    # 패치한다 - spai_runner.py는 `from app.services.gpu_queue import get_gpu_queue`로
    # 가져다 쓰므로(다른 테스트가 get_settings를 패치하는 것과 동일한 이유), 소스 모듈 쪽을
    # 패치하면 이미 바인딩된 spai_runner.get_gpu_queue에는 반영되지 않는다.
    monkeypatch.setattr("app.services.spai_runner.get_gpu_queue", lambda: FakeQueue())

    mock_get_settings.return_value = _spai_settings(tmp_path)

    def fake_run(command, **kwargs):
        output_dir = Path(command[command.index("--output") + 1])
        _write_score_csv(output_dir, "0.5")
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    from app.services.spai_runner import run_spai_inference

    run_spai_inference(b"fake", "test.jpg")

    assert calls == ["spai"]
