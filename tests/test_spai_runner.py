import base64
import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.spai_runner import _find_and_encode_overlay, _safe_filename


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
