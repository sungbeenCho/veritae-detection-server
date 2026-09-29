import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.misinfo_runner import MisinfoInferenceError, run_misinfo_inference


@pytest.fixture(autouse=True)
def _stub_gpu_queue(monkeypatch):
    # test_scam_runner.py/test_spai_runner.py/test_dfdc_runner.py의 _stub_gpu_queue와
    # 동일한 이유 - misinfo_infer는 CPU 전용 경로 없이 항상 get_gpu_queue().acquire(...)를
    # 거치므로, 큐 사용 자체를 검증하지 않는 테스트들도 get_gpu_queue를 거친다. 그대로 두면
    # get_settings만 목킹된 상태에서 실제 GpuQueue.get_gpu_queue()가 호출돼 SPAI_REPO_DIR 등을
    # 요구하는 진짜 Settings()가 생성되면서 무관한 RuntimeError로 깨진다. 큐 동작 자체를
    # 검증하는 테스트는 이 기본값을 자기 안에서 다시 덮어쓴다.
    from contextlib import nullcontext

    class _NoopGpuQueue:
        def acquire(self, label):
            return nullcontext()

    monkeypatch.setattr("app.services.misinfo_runner.get_gpu_queue", lambda: _NoopGpuQueue())


def _misinfo_settings(tmp_path) -> MagicMock:
    settings = MagicMock()
    settings.misinfo_script = Path("/fake/misinfo_infer.py")
    settings.text_extraction_python = "python"
    settings.text_extraction_work_dir = tmp_path
    settings.wiki_index_path = Path("/fake/wiki_index.sqlite3")
    settings.ollama_url = "http://localhost:11434"
    settings.ollama_model = "qwen3.5:4b"
    settings.misinfo_evidence_chunk_count = 5
    settings.misinfo_timeout_seconds = 300
    return settings


def _write_result_json(output_file: Path, claims=None, wiki_snapshot="2026-09-01") -> None:
    output_file.write_text(
        json.dumps({"model": "qwen3.5:4b", "wiki_snapshot": wiki_snapshot, "claims": claims or []}),
        encoding="utf-8",
    )


@patch("app.services.misinfo_runner.subprocess.run")
@patch("app.services.misinfo_runner.get_settings")
def test_run_misinfo_inference_returns_claims(mock_get_settings, mock_run, tmp_path):
    mock_get_settings.return_value = _misinfo_settings(tmp_path)

    def fake_run(command, **kwargs):
        output_file = Path(command[command.index("--output") + 1])
        _write_result_json(
            output_file,
            claims=[{"sentence": "선풍기를 틀고 자면 사망한다.", "reason": "미신이다.", "evidence": []}],
        )
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    result = run_misinfo_inference(["선풍기를 틀고 자면 사망한다."])

    assert result.wiki_snapshot == "2026-09-01"
    assert result.claims == [{"sentence": "선풍기를 틀고 자면 사망한다.", "reason": "미신이다.", "evidence": []}]
    called_command = mock_run.call_args.args[0]
    assert "--sentences" in called_command


@patch("app.services.misinfo_runner.subprocess.run")
@patch("app.services.misinfo_runner.get_settings")
def test_run_misinfo_inference_raises_on_nonzero_exit(mock_get_settings, mock_run, tmp_path):
    mock_get_settings.return_value = _misinfo_settings(tmp_path)
    mock_run.return_value = MagicMock(returncode=1, stderr="boom")

    with pytest.raises(MisinfoInferenceError):
        run_misinfo_inference(["아무 문장"])


@patch("app.services.misinfo_runner.subprocess.run")
@patch("app.services.misinfo_runner.get_settings")
def test_run_misinfo_inference_raises_on_timeout(mock_get_settings, mock_run, tmp_path):
    import subprocess

    mock_get_settings.return_value = _misinfo_settings(tmp_path)
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="misinfo_infer.py", timeout=300)

    with pytest.raises(MisinfoInferenceError):
        run_misinfo_inference(["아무 문장"])


@patch("app.services.misinfo_runner.subprocess.run")
@patch("app.services.misinfo_runner.get_settings")
def test_run_misinfo_inference_uses_gpu_queue(mock_get_settings, mock_run, tmp_path, monkeypatch):
    mock_get_settings.return_value = _misinfo_settings(tmp_path)
    calls = []

    class FakeQueue:
        def acquire(self, label):
            calls.append(label)
            from contextlib import contextmanager

            @contextmanager
            def cm():
                yield

            return cm()

    monkeypatch.setattr("app.services.misinfo_runner.get_gpu_queue", lambda: FakeQueue())

    def fake_run(command, **kwargs):
        output_file = Path(command[command.index("--output") + 1])
        _write_result_json(output_file)
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    run_misinfo_inference(["아무 문장"])

    assert calls == ["misinfo_infer"]
