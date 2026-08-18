import json

import pytest

from app.services.antideepfake_runner import (
    AntiDeepfakeInferenceError,
    _parse_result,
    _safe_filename,
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
