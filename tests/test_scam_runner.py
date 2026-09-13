import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.scam_runner import (
    ScamInferenceError,
    _safe_filename,
    run_scam_inference_audio,
    run_scam_inference_image,
    run_scam_inference_video,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("photo.jpg", "photo.jpg"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\Windows\\System32\\evil.dll", "evil.dll"),
        ("..", "upload"),
        ("", "upload"),
    ],
)
def test_safe_filename_strips_path_traversal(raw, expected):
    assert _safe_filename(raw) == expected


def _scam_settings(tmp_path) -> MagicMock:
    settings = MagicMock()
    settings.text_extraction_python = "python"
    settings.text_extraction_script = Path("/fake/scam_infer.py")
    settings.text_extraction_timeout_seconds = 300
    settings.text_extraction_work_dir = tmp_path
    settings.lilju_model_id = "Lilju/voicephishing_kobert"
    settings.paddleocr_lang = "korean"
    settings.whisper_model_size = "large-v3"
    return settings


def _write_result_json(output_file: Path, score, evidence=None) -> None:
    output_file.write_text(json.dumps({"score": score, "evidence": evidence or []}), encoding="utf-8")


@patch("app.services.scam_runner.subprocess.run")
@patch("app.services.scam_runner.get_settings")
def test_run_scam_inference_image_returns_score_and_evidence(mock_get_settings, mock_run, tmp_path):
    mock_get_settings.return_value = _scam_settings(tmp_path)

    def fake_run(command, **kwargs):
        output_file = Path(command[command.index("--output") + 1])
        _write_result_json(output_file, 0.82, [{"sentence": "계좌번호를 알려주세요", "score": 0.95}])
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    result = run_scam_inference_image(b"fake-image-bytes", "test.jpg")

    assert result.score == 0.82
    assert result.evidence == [{"sentence": "계좌번호를 알려주세요", "score": 0.95}]
    called_command = mock_run.call_args.args[0]
    assert called_command[called_command.index("--mode") + 1] == "ocr"


@patch("app.services.scam_runner.subprocess.run")
@patch("app.services.scam_runner.get_settings")
def test_run_scam_inference_audio_uses_stt_mode_and_returns_none_score_when_no_text(mock_get_settings, mock_run, tmp_path):
    mock_get_settings.return_value = _scam_settings(tmp_path)

    def fake_run(command, **kwargs):
        output_file = Path(command[command.index("--output") + 1])
        _write_result_json(output_file, None)
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    result = run_scam_inference_audio(b"fake-audio-bytes", "test.wav")

    assert result.score is None
    assert result.evidence == []
    called_command = mock_run.call_args.args[0]
    assert called_command[called_command.index("--mode") + 1] == "stt"


@patch("app.services.scam_runner.subprocess.run")
@patch("app.services.scam_runner.get_settings")
def test_run_scam_inference_raises_on_nonzero_exit(mock_get_settings, mock_run, tmp_path):
    mock_get_settings.return_value = _scam_settings(tmp_path)
    mock_run.return_value = MagicMock(returncode=1, stderr="boom")

    with pytest.raises(ScamInferenceError):
        run_scam_inference_image(b"fake-image-bytes", "test.jpg")


@patch("app.services.scam_runner.subprocess.run")
@patch("app.services.scam_runner.get_settings")
def test_run_scam_inference_raises_on_timeout(mock_get_settings, mock_run, tmp_path):
    import subprocess

    mock_get_settings.return_value = _scam_settings(tmp_path)
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="scam_infer.py", timeout=300)

    with pytest.raises(ScamInferenceError):
        run_scam_inference_image(b"fake-image-bytes", "test.jpg")


@patch("app.services.scam_runner.subprocess.run")
@patch("app.services.scam_runner.get_settings")
def test_run_scam_inference_video_extracts_audio_then_runs_stt(mock_get_settings, mock_run, tmp_path):
    mock_get_settings.return_value = _scam_settings(tmp_path)

    def fake_run(command, **kwargs):
        if command[0] == "ffmpeg":
            audio_path = Path(command[-1])
            audio_path.write_bytes(b"fake-wav-bytes")
            return MagicMock(returncode=0, stderr="")
        output_file = Path(command[command.index("--output") + 1])
        _write_result_json(output_file, 0.6, [{"sentence": "지금 바로 이체하세요", "score": 0.9}])
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    result = run_scam_inference_video(b"fake-video-bytes", "test.mp4")

    assert result.score == 0.6
    assert mock_run.call_count == 2
    first_command = mock_run.call_args_list[0].args[0]
    assert first_command[0] == "ffmpeg"
    second_command = mock_run.call_args_list[1].args[0]
    assert second_command[second_command.index("--mode") + 1] == "stt"


@patch("app.services.scam_runner.subprocess.run")
@patch("app.services.scam_runner.get_settings")
def test_run_scam_inference_video_raises_when_ffmpeg_fails(mock_get_settings, mock_run, tmp_path):
    mock_get_settings.return_value = _scam_settings(tmp_path)
    mock_run.return_value = MagicMock(returncode=1, stderr="ffmpeg boom")

    with pytest.raises(ScamInferenceError):
        run_scam_inference_video(b"fake-video-bytes", "test.mp4")
