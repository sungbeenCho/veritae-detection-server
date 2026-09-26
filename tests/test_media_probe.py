import subprocess
from unittest.mock import MagicMock, patch

from app.services.media_probe import probe_audio_duration_seconds


@patch("app.services.media_probe.subprocess.run")
def test_probe_audio_duration_returns_seconds_on_normal_output(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="301.234000\n", stderr="")

    duration = probe_audio_duration_seconds(b"fake-audio-bytes", "test.wav")

    assert duration == 301.234


@patch("app.services.media_probe.subprocess.run")
def test_probe_audio_duration_returns_none_when_output_is_not_a_number(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="N/A\n", stderr="")

    duration = probe_audio_duration_seconds(b"fake-audio-bytes", "test.wav")

    assert duration is None


@patch("app.services.media_probe.subprocess.run")
def test_probe_audio_duration_returns_none_on_nonzero_exit(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="boom")

    duration = probe_audio_duration_seconds(b"fake-audio-bytes", "test.wav")

    assert duration is None


@patch("app.services.media_probe.subprocess.run")
def test_probe_audio_duration_returns_none_on_timeout(mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

    duration = probe_audio_duration_seconds(b"fake-audio-bytes", "test.wav")

    assert duration is None
