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
    # 실제로 존재하는 모델 폴더를 가리켜야 한다 - 러너가 요청 시작 시 폴더를 검사한다.
    nli_model = tmp_path / "nli_model_aug"
    nli_model.mkdir()
    (nli_model / "config.json").write_text("{}", encoding="utf-8")
    settings.misinfo_nli_model = nli_model
    settings.misinfo_wiki_search = False
    settings.misinfo_judge_mode = "current"
    settings.misinfo_speaker_model = tmp_path / "speaker_model_v2"
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
    assert called_command[called_command.index("--nli-model") + 1] == str(mock_get_settings.return_value.misinfo_nli_model)
    assert "--wiki-search" not in called_command  # 외부 전송이라 기본은 꺼짐


@patch("app.services.misinfo_runner.subprocess.run")
@patch("app.services.misinfo_runner.get_settings")
def test_run_misinfo_inference_logs_subprocess_log_even_on_success(mock_get_settings, mock_run, tmp_path, caplog):
    """하위 프로세스의 단계별 시간·경고(LLM 형식 오류 등)는 stderr로 나온다. 성공하면 버려져서 서버 로그에서
    판정이 왜 비었는지 알 수 없었다(2026-10-13)."""
    import logging

    mock_get_settings.return_value = _misinfo_settings(tmp_path)

    def fake_run(command, **kwargs):
        _write_result_json(Path(command[command.index("--output") + 1]))
        return MagicMock(returncode=0, stderr="sentence 1/2 total=3.2s outcome=not_refuted")

    mock_run.side_effect = fake_run

    with caplog.at_level(logging.INFO, logger="app.services.misinfo_runner"):
        run_misinfo_inference(["문장"])

    assert "sentence 1/2 total=3.2s outcome=not_refuted" in caplog.text


@patch("app.services.misinfo_runner.subprocess.run")
@patch("app.services.misinfo_runner.get_settings")
def test_run_misinfo_inference_forces_utf8_output_from_subprocess(mock_get_settings, mock_run, tmp_path):
    """한글 로그를 utf-8로 읽으므로 하위 프로세스도 utf-8로 쓰게 한다(scam_runner와 같은 이유, Windows 콘솔은 cp949)."""
    mock_get_settings.return_value = _misinfo_settings(tmp_path)

    def fake_run(command, **kwargs):
        _write_result_json(Path(command[command.index("--output") + 1]))
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    run_misinfo_inference(["문장"])

    assert mock_run.call_args.kwargs["env"]["PYTHONIOENCODING"] == "utf-8"


@patch("app.services.misinfo_runner.subprocess.run")
@patch("app.services.misinfo_runner.get_settings")
def test_run_misinfo_inference_turns_off_model_loading_progress_bars(mock_get_settings, mock_run, tmp_path):
    """모델 로드 진행 막대("Loading weights: 100%|███")가 서버 로그 줄을 어지럽힌다(2026-10-13 실제 실행에서 확인)."""
    mock_get_settings.return_value = _misinfo_settings(tmp_path)

    def fake_run(command, **kwargs):
        _write_result_json(Path(command[command.index("--output") + 1]))
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    run_misinfo_inference(["문장"])

    env = mock_run.call_args.kwargs["env"]
    assert env["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert env["TRANSFORMERS_NO_PROGRESS_BAR"] == "1"


@patch("app.services.misinfo_runner.subprocess.run")
@patch("app.services.misinfo_runner.get_settings")
def test_run_misinfo_inference_passes_wiki_search_only_when_enabled(mock_get_settings, mock_run, tmp_path):
    settings = _misinfo_settings(tmp_path)
    settings.misinfo_wiki_search = True
    mock_get_settings.return_value = settings

    def fake_run(command, **kwargs):
        _write_result_json(Path(command[command.index("--output") + 1]))
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    run_misinfo_inference(["문장"])

    assert "--wiki-search" in mock_run.call_args.args[0]


@patch("app.services.misinfo_runner.subprocess.run")
@patch("app.services.misinfo_runner.get_settings")
def test_run_misinfo_inference_fails_fast_when_nli_model_folder_is_missing(mock_get_settings, mock_run, tmp_path):
    """NLI 모델은 반박이 나온 문장에서 처음 읽는다. 폴더가 없으면 요청 도중에, 반박이 있을 때만 늦게 실패하므로
    (불규칙해서 원인 찾기 어려움) 요청 시작 시점에 경로가 담긴 명확한 오류로 바로 실패해야 한다."""
    settings = _misinfo_settings(tmp_path)
    settings.misinfo_nli_model = tmp_path / "없는_nli_폴더"
    mock_get_settings.return_value = settings

    def fake_run(command, **kwargs):
        _write_result_json(Path(command[command.index("--output") + 1]))
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    with pytest.raises(MisinfoInferenceError, match="없는_nli_폴더"):
        run_misinfo_inference(["아무 문장"])

    mock_run.assert_not_called()


@patch("app.services.misinfo_runner.subprocess.run")
@patch("app.services.misinfo_runner.get_settings")
def test_run_misinfo_inference_fails_fast_when_nli_folder_has_no_model(mock_get_settings, mock_run, tmp_path):
    """폴더만 있고 모델 파일(config.json)이 없는 빈 폴더도 같은 방식으로 막는다."""
    settings = _misinfo_settings(tmp_path)
    empty = tmp_path / "빈_nli_폴더"
    empty.mkdir()
    settings.misinfo_nli_model = empty
    mock_get_settings.return_value = settings

    with pytest.raises(MisinfoInferenceError, match="빈_nli_폴더"):
        run_misinfo_inference(["아무 문장"])

    mock_run.assert_not_called()


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


@patch("app.services.misinfo_runner.subprocess.run")
@patch("app.services.misinfo_runner.get_settings")
def test_split_mode_passes_judge_mode_and_speaker_model(mock_get_settings, mock_run, tmp_path):
    """나눔+A(split)면 판정 방식과 말한 사람 판별 모델 폴더를 하위 프로세스에 넘긴다."""
    settings = _misinfo_settings(tmp_path)
    settings.misinfo_judge_mode = "split"
    settings.misinfo_speaker_model.mkdir()
    (settings.misinfo_speaker_model / "config.json").write_text("{}", encoding="utf-8")
    mock_get_settings.return_value = settings

    def fake_run(command, **kwargs):
        _write_result_json(Path(command[command.index("--output") + 1]))
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run
    run_misinfo_inference(["문장"])
    command = mock_run.call_args[0][0]
    assert command[command.index("--judge-mode") + 1] == "split"
    assert command[command.index("--speaker-model") + 1] == str(settings.misinfo_speaker_model)


@patch("app.services.misinfo_runner.subprocess.run")
@patch("app.services.misinfo_runner.get_settings")
def test_split_mode_without_speaker_model_fails_early(mock_get_settings, mock_run, tmp_path):
    settings = _misinfo_settings(tmp_path)
    settings.misinfo_judge_mode = "split"
    mock_get_settings.return_value = settings

    with pytest.raises(MisinfoInferenceError, match="말한 사람 판별 모델"):
        run_misinfo_inference(["문장"])
    mock_run.assert_not_called()
