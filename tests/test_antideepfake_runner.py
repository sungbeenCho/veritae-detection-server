import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.services.antideepfake_runner import (
    AntiDeepfakeInferenceError,
    _parse_result,
    _safe_filename,
    run_antideepfake_inference,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("voice.wav", "voice.wav"),
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


def test_parse_result_reads_score_and_evidence(tmp_path):
    output_file = tmp_path / "result.json"
    output_file.write_text(
        json.dumps(
            {
                "score": 0.87,
                "evidence": [
                    {
                        "title": "시간 구간 이상 패턴",
                        "description": "설명",
                        "tags": ["temporal"],
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                    }
                ],
            }
        )
    )

    result = _parse_result(output_file)

    assert result.score == 0.87
    assert result.evidence[0]["title"] == "시간 구간 이상 패턴"


def test_parse_result_missing_score_raises(tmp_path):
    output_file = tmp_path / "result.json"
    output_file.write_text(json.dumps({"evidence": []}))

    with pytest.raises(AntiDeepfakeInferenceError):
        _parse_result(output_file)


def test_parse_result_corrupted_json_raises(tmp_path):
    output_file = tmp_path / "result.json"
    output_file.write_text("not valid json")

    with pytest.raises(AntiDeepfakeInferenceError):
        _parse_result(output_file)


def test_run_antideepfake_inference_passes_repo_dir_as_cwd(monkeypatch, tmp_path):
    # Regression test: the antideepfake subprocess must run with cwd set to the
    # AntiDeepfake repo dir, mirroring how spai_runner.py invokes its subprocess
    # (cwd=settings.spai_repo_dir). Without this, any relative-path file access
    # inside AntiDeepfake's own library code would resolve against the wrong dir.
    monkeypatch.setenv("SPAI_REPO_DIR", str(tmp_path / "spai"))
    monkeypatch.setenv("ANTIDEEPFAKE_REPO_DIR", str(tmp_path / "antideepfake"))
    monkeypatch.setenv("ANTIDEEPFAKE_WORK_DIR", str(tmp_path / "work"))
    get_settings.cache_clear()
    settings = get_settings()

    captured_kwargs = {}

    def fake_run(command, **kwargs):
        captured_kwargs.update(kwargs)
        output_file = Path(command[command.index("--output") + 1])
        output_file.write_text(json.dumps({"score": 0.42, "evidence": []}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("app.services.antideepfake_runner.subprocess.run", fake_run)

    try:
        result = run_antideepfake_inference(b"fake-audio-bytes", "voice.wav")
    finally:
        get_settings.cache_clear()

    assert result.score == 0.42
    assert captured_kwargs.get("cwd") == settings.antideepfake_repo_dir
