import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from app.services.dfdc_runner import DfdcInferenceError, _parse_result, _safe_filename


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("video.mp4", "video.mp4"),
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


def test_parse_result_with_valid_json(tmp_path):
    output_file = tmp_path / "result.json"
    output_file.write_text(
        json.dumps({"score": 0.91, "evidence": [{"title": "x"}], "evidence_image": "base64data"}),
        encoding="utf-8",
    )

    result = _parse_result(output_file)

    assert result.score == 0.91
    assert result.evidence == [{"title": "x"}]
    assert result.evidence_image == "base64data"


def test_parse_result_with_missing_evidence_image_defaults_to_none(tmp_path):
    output_file = tmp_path / "result.json"
    output_file.write_text(json.dumps({"score": 0.1, "evidence": []}), encoding="utf-8")

    result = _parse_result(output_file)

    assert result.evidence_image is None


def test_parse_result_missing_score_raises(tmp_path):
    output_file = tmp_path / "result.json"
    output_file.write_text(json.dumps({"evidence": []}), encoding="utf-8")

    with pytest.raises(DfdcInferenceError, match="missing 'score'"):
        _parse_result(output_file)


def test_parse_result_invalid_json_raises(tmp_path):
    output_file = tmp_path / "result.json"
    output_file.write_text("not json", encoding="utf-8")

    with pytest.raises(DfdcInferenceError):
        _parse_result(output_file)


@patch("app.services.dfdc_runner.subprocess.run")
@patch("app.services.dfdc_runner.get_settings")
def test_run_dfdc_inference_success(mock_get_settings, mock_run, tmp_path):
    from app.services.dfdc_runner import run_dfdc_inference

    settings = MagicMock()
    settings.dfdc_work_dir = tmp_path
    settings.dfdc_python = "python"
    settings.dfdc_script = tmp_path / "dfdc_infer.py"
    settings.dfdc_repo_dir = tmp_path
    settings.dfdc_checkpoint = tmp_path / "checkpoint"
    settings.dfdc_timeout_seconds = 600
    mock_get_settings.return_value = settings

    def fake_run(command, **kwargs):
        output_path = command[command.index("--output") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"score": 0.91, "evidence": [], "evidence_image": None}, f)
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    result = run_dfdc_inference(b"fake-video-bytes", "test.mp4")

    assert result.score == 0.91


@patch("app.services.dfdc_runner.subprocess.run")
@patch("app.services.dfdc_runner.get_settings")
def test_run_dfdc_inference_nonzero_exit_raises(mock_get_settings, mock_run, tmp_path):
    from app.services.dfdc_runner import run_dfdc_inference

    settings = MagicMock()
    settings.dfdc_work_dir = tmp_path
    settings.dfdc_python = "python"
    settings.dfdc_script = tmp_path / "dfdc_infer.py"
    settings.dfdc_repo_dir = tmp_path
    settings.dfdc_checkpoint = tmp_path / "checkpoint"
    settings.dfdc_timeout_seconds = 600
    mock_get_settings.return_value = settings
    mock_run.return_value = MagicMock(returncode=1, stderr="CUDA out of memory")

    with pytest.raises(DfdcInferenceError, match="CUDA out of memory"):
        run_dfdc_inference(b"fake-video-bytes", "test.mp4")


@patch("app.services.dfdc_runner.subprocess.run")
@patch("app.services.dfdc_runner.get_settings")
def test_run_dfdc_inference_timeout_raises(mock_get_settings, mock_run, tmp_path):
    from app.services.dfdc_runner import run_dfdc_inference

    settings = MagicMock()
    settings.dfdc_work_dir = tmp_path
    settings.dfdc_python = "python"
    settings.dfdc_script = tmp_path / "dfdc_infer.py"
    settings.dfdc_repo_dir = tmp_path
    settings.dfdc_checkpoint = tmp_path / "checkpoint"
    settings.dfdc_timeout_seconds = 600
    mock_get_settings.return_value = settings
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="dfdc_infer.py", timeout=600)

    with pytest.raises(DfdcInferenceError, match="timed out"):
        run_dfdc_inference(b"fake-video-bytes", "test.mp4")
