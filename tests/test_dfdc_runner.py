import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.config import get_settings
from app.services.dfdc_runner import DfdcInferenceError, NoFaceDetectedError, _parse_result, _safe_filename


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
    settings.dfdc_checkpoints = [tmp_path / "checkpoint1", tmp_path / "checkpoint2"]
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
def test_run_dfdc_inference_passes_all_checkpoints_to_script(mock_get_settings, mock_run, tmp_path):
    # Regression test: 7개 체크포인트 앙상블 전부가 --checkpoints 뒤에 개별 인자로 붙어야
    # dfdc_infer.py의 argparse(nargs="+")가 전부 받는다 - 예전에 단일 체크포인트로 축소했다가
    # 근거 없는 GPU 메모리 우려 때문이었던 걸로 밝혀져 원복(config.py 참고).
    from app.services.dfdc_runner import run_dfdc_inference

    settings = MagicMock()
    settings.dfdc_work_dir = tmp_path
    settings.dfdc_python = "python"
    settings.dfdc_script = tmp_path / "dfdc_infer.py"
    settings.dfdc_repo_dir = tmp_path
    settings.dfdc_checkpoints = [tmp_path / f"checkpoint{i}" for i in range(7)]
    settings.dfdc_timeout_seconds = 600
    mock_get_settings.return_value = settings

    captured_command = {}

    def fake_run(command, **kwargs):
        captured_command["value"] = command
        output_path = command[command.index("--output") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"score": 0.91, "evidence": [], "evidence_image": None}, f)
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    run_dfdc_inference(b"fake-video-bytes", "test.mp4")

    command = captured_command["value"]
    checkpoints_idx = command.index("--checkpoints")
    input_idx = command.index("--input")
    passed_checkpoints = command[checkpoints_idx + 1:input_idx]
    assert passed_checkpoints == [str(c) for c in settings.dfdc_checkpoints]


@patch("app.services.dfdc_runner.subprocess.run")
@patch("app.services.dfdc_runner.get_settings")
def test_run_dfdc_inference_nonzero_exit_raises(mock_get_settings, mock_run, tmp_path):
    from app.services.dfdc_runner import run_dfdc_inference

    settings = MagicMock()
    settings.dfdc_work_dir = tmp_path
    settings.dfdc_python = "python"
    settings.dfdc_script = tmp_path / "dfdc_infer.py"
    settings.dfdc_repo_dir = tmp_path
    settings.dfdc_checkpoints = [tmp_path / "checkpoint1", tmp_path / "checkpoint2"]
    settings.dfdc_timeout_seconds = 600
    mock_get_settings.return_value = settings
    mock_run.return_value = MagicMock(returncode=1, stderr="CUDA out of memory")

    with pytest.raises(DfdcInferenceError, match="CUDA out of memory"):
        run_dfdc_inference(b"fake-video-bytes", "test.mp4")


@patch("app.services.dfdc_runner.subprocess.run")
@patch("app.services.dfdc_runner.get_settings")
def test_run_dfdc_inference_exit_code_2_raises_no_face_detected(mock_get_settings, mock_run, tmp_path):
    # Regression test: dfdc_infer.py는 얼굴을 못 찾으면 종료 코드 2로 끝난다(다른 실패는 1) -
    # video.py가 422(사용자 케이스)와 502(서버 장애)를 구분해 응답할 수 있어야 하므로,
    # 이 신호가 여기서 NoFaceDetectedError로 정확히 갈라져야 한다(2026-08-27).
    from app.services.dfdc_runner import run_dfdc_inference

    settings = MagicMock()
    settings.dfdc_work_dir = tmp_path
    settings.dfdc_python = "python"
    settings.dfdc_script = tmp_path / "dfdc_infer.py"
    settings.dfdc_repo_dir = tmp_path
    settings.dfdc_checkpoints = [tmp_path / "checkpoint1", tmp_path / "checkpoint2"]
    settings.dfdc_timeout_seconds = 600
    mock_get_settings.return_value = settings
    mock_run.return_value = MagicMock(returncode=2, stderr="얼굴을 찾을 수 없습니다.")

    with pytest.raises(NoFaceDetectedError, match="얼굴을 찾을 수 없습니다"):
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
    settings.dfdc_checkpoints = [tmp_path / "checkpoint1", tmp_path / "checkpoint2"]
    settings.dfdc_timeout_seconds = 600
    mock_get_settings.return_value = settings
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="dfdc_infer.py", timeout=600)

    with pytest.raises(DfdcInferenceError, match="timed out"):
        run_dfdc_inference(b"fake-video-bytes", "test.mp4")


def test_run_dfdc_inference_passes_repo_dir_as_cwd(monkeypatch, tmp_path):
    # Regression test: DFDC_CHECKPOINT의 기본값이 상대경로(./weights/...)라서, subprocess
    # 호출 시 cwd=settings.dfdc_repo_dir가 반드시 지정돼야 체크포인트를 찾을 수 있다.
    # test_antideepfake_runner.py의 test_run_antideepfake_inference_passes_repo_dir_as_cwd와
    # 동일한 패턴 - antideepfake_runner.py/spai_runner.py도 같은 방식으로 cwd를 설정한다.
    monkeypatch.setenv("SPAI_REPO_DIR", str(tmp_path / "spai"))
    monkeypatch.setenv("ANTIDEEPFAKE_REPO_DIR", str(tmp_path / "antideepfake"))
    monkeypatch.setenv("DFDC_REPO_DIR", str(tmp_path / "dfdc"))
    monkeypatch.setenv("DFDC_WORK_DIR", str(tmp_path / "work"))
    get_settings.cache_clear()
    settings = get_settings()

    captured_kwargs = {}

    def fake_run(command, **kwargs):
        captured_kwargs.update(kwargs)
        output_file = Path(command[command.index("--output") + 1])
        output_file.write_text(json.dumps({"score": 0.42, "evidence": [], "evidence_image": None}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("app.services.dfdc_runner.subprocess.run", fake_run)

    from app.services.dfdc_runner import run_dfdc_inference

    try:
        result = run_dfdc_inference(b"fake-video-bytes", "test.mp4")
    finally:
        get_settings.cache_clear()

    assert result.score == 0.42
    assert captured_kwargs.get("cwd") == settings.dfdc_repo_dir
