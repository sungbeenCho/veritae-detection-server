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
def test_run_scam_inference_video_extracts_audio_and_frames_then_runs_video_mode(mock_get_settings, mock_run, tmp_path):
    mock_get_settings.return_value = _scam_settings(tmp_path)

    def fake_run(command, **kwargs):
        if command[0] == "ffmpeg" and "-acodec" in command:
            audio_path = Path(command[-1])
            audio_path.write_bytes(b"fake-wav-bytes")
            return MagicMock(returncode=0, stderr="")
        if command[0] == "ffmpeg":
            # 프레임 추출 호출 - 실제 프레임 파일까지는 안 만들어도 이 테스트엔 지장 없음
            return MagicMock(returncode=0, stderr="")
        output_file = Path(command[command.index("--output") + 1])
        _write_result_json(output_file, 0.6, [{"sentence": "지금 바로 이체하세요", "score": 0.9}])
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    result = run_scam_inference_video(b"fake-video-bytes", "test.mp4")

    assert result.score == 0.6
    assert mock_run.call_count == 3
    audio_command = mock_run.call_args_list[0].args[0]
    assert audio_command[0] == "ffmpeg"
    assert "-acodec" in audio_command
    frame_command = mock_run.call_args_list[1].args[0]
    assert frame_command[0] == "ffmpeg"
    assert "-vf" in frame_command
    infer_command = mock_run.call_args_list[2].args[0]
    assert infer_command[infer_command.index("--mode") + 1] == "video"
    assert "--frames-dir" in infer_command


@patch("app.services.scam_runner.subprocess.run")
@patch("app.services.scam_runner.get_settings")
def test_run_scam_inference_video_omits_frames_dir_when_frame_extraction_fails(mock_get_settings, mock_run, tmp_path):
    """프레임 추출이 실패해도 전체 요청은 실패시키지 않는다 - best-effort로 음성만
    가지고 진행한다(scamDetection이 아예 null이 되는 것과는 다른 케이스: 이건 오디오
    추출 자체는 성공했을 때의 얘기)."""
    mock_get_settings.return_value = _scam_settings(tmp_path)

    def fake_run(command, **kwargs):
        if command[0] == "ffmpeg" and "-acodec" in command:
            audio_path = Path(command[-1])
            audio_path.write_bytes(b"fake-wav-bytes")
            return MagicMock(returncode=0, stderr="")
        if command[0] == "ffmpeg":
            return MagicMock(returncode=1, stderr="frame extraction boom")
        output_file = Path(command[command.index("--output") + 1])
        _write_result_json(output_file, 0.6, [{"sentence": "지금 바로 이체하세요", "score": 0.9}])
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    result = run_scam_inference_video(b"fake-video-bytes", "test.mp4")

    assert result.score == 0.6
    infer_command = mock_run.call_args_list[2].args[0]
    assert "--frames-dir" not in infer_command


@patch("app.services.scam_runner.subprocess.run")
@patch("app.services.scam_runner.get_settings")
def test_run_scam_inference_video_omits_audio_when_no_audio_track(mock_get_settings, mock_run, tmp_path):
    """음성 트랙이 없는 영상(무음 데모/샘플 영상 등)은 정상적인 케이스지 장애가 아니다 -
    프레임 추출 실패 처리와 동일하게, 음성 추출이 실패해도 화면 텍스트만으로 best-effort
    진행한다(2026-09-22, 3060Ti 실기에서 발견된 버그 - sample_deepfake.mp4가 무음이라
    이전엔 전체 요청이 502로 실패했었음)."""
    mock_get_settings.return_value = _scam_settings(tmp_path)

    def fake_run(command, **kwargs):
        if command[0] == "ffmpeg" and "-acodec" in command:
            return MagicMock(returncode=1, stderr="Output file does not contain any stream")
        if command[0] == "ffmpeg":
            return MagicMock(returncode=0, stderr="")
        output_file = Path(command[command.index("--output") + 1])
        _write_result_json(output_file, 0.6, [{"sentence": "지금 바로 이체하세요", "score": 0.9}])
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    result = run_scam_inference_video(b"fake-video-bytes", "test.mp4")

    assert result.score == 0.6
    infer_command = mock_run.call_args_list[-1].args[0]
    assert "--input" not in infer_command
    assert "--frames-dir" in infer_command


@patch("app.services.scam_runner.subprocess.run")
@patch("app.services.scam_runner.get_settings")
def test_run_scam_inference_video_returns_empty_when_audio_and_frames_both_unavailable(
    mock_get_settings, mock_run, tmp_path
):
    """음성도 화면 텍스트도 둘 다 추출 실패해도 예외가 아니라 결과없음(null)으로 끝난다 -
    실제로 분석할 게 없는 것뿐이지 파이프라인 장애가 아니다."""
    mock_get_settings.return_value = _scam_settings(tmp_path)

    def fake_run(command, **kwargs):
        if command[0] == "ffmpeg":
            return MagicMock(returncode=1, stderr="boom")
        output_file = Path(command[command.index("--output") + 1])
        _write_result_json(output_file, None, [])
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    result = run_scam_inference_video(b"fake-video-bytes", "test.mp4")

    assert result.score is None
    infer_command = mock_run.call_args_list[-1].args[0]
    assert "--input" not in infer_command
    assert "--frames-dir" not in infer_command


@patch("app.services.scam_runner.subprocess.run")
@patch("app.services.scam_runner.get_settings")
def test_run_scam_inference_video_raises_when_inference_itself_fails(mock_get_settings, mock_run, tmp_path):
    """ffmpeg 추출(음성/프레임)이 아니라 실제 추론 스크립트(scam_infer.py) 자체가 실패하면
    이건 진짜 장애라 여전히 예외를 던져야 한다."""
    mock_get_settings.return_value = _scam_settings(tmp_path)

    def fake_run(command, **kwargs):
        if command[0] == "ffmpeg" and "-acodec" in command:
            audio_path = Path(command[-1])
            audio_path.write_bytes(b"fake-wav-bytes")
            return MagicMock(returncode=0, stderr="")
        if command[0] == "ffmpeg":
            return MagicMock(returncode=0, stderr="")
        return MagicMock(returncode=1, stderr="inference boom")

    mock_run.side_effect = fake_run

    with pytest.raises(ScamInferenceError):
        run_scam_inference_video(b"fake-video-bytes", "test.mp4")
