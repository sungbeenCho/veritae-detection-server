from contextlib import contextmanager

import pytest

from app.services import text_judge
from app.services.misinfo_runner import MisinfoInferenceError, MisinfoResult
from app.services.scam_judge import ScamJudgeError, ScamJudgment

SCAM = ScamJudgment(model="qwen3.5:4b-scam-v3", score=0.9, reason="이유")
MISINFO = MisinfoResult(model="qwen3.5:4b", wiki_snapshot="2026-09-01", claims=[])


@pytest.fixture
def events(monkeypatch):
    """대기열 자리 잡기/놓기와 각 판정·모델 내리기가 일어난 순서를 기록한다."""
    log: list[str] = []

    class RecordingQueue:
        @contextmanager
        def acquire(self, label):
            log.append(f"acquire:{label}")
            try:
                yield
            finally:
                log.append("release")

    monkeypatch.setattr(text_judge, "get_gpu_queue", lambda: RecordingQueue())
    monkeypatch.setattr(text_judge, "unload_ollama_model", lambda: log.append("unload"))
    return log


def test_scam_and_misinfo_run_back_to_back_in_one_queue_slot_then_model_unloads(events, monkeypatch):
    # 자리를 놓기 전에 모델을 내려야, 다음 요청의 Whisper가 Ollama와 같이 GPU에 올라가지 않는다.
    monkeypatch.setattr(text_judge, "judge_scam", lambda sentences: events.append("scam") or SCAM)
    monkeypatch.setattr(text_judge, "run_misinfo_inference", lambda sentences: events.append("misinfo") or MISINFO)

    result = text_judge.run_text_judgments(["문장"])

    assert events == ["acquire:text_judge", "scam", "misinfo", "unload", "release"]
    assert result.scam is SCAM
    assert result.misinfo is MISINFO


def test_model_unloads_even_when_scam_judgment_fails(events, monkeypatch):
    def fail(sentences):
        raise ScamJudgeError("boom")

    monkeypatch.setattr(text_judge, "judge_scam", fail)
    monkeypatch.setattr(text_judge, "run_misinfo_inference", lambda sentences: events.append("misinfo") or MISINFO)

    with pytest.raises(ScamJudgeError):
        text_judge.run_text_judgments(["문장"])

    assert events == ["acquire:text_judge", "unload", "release"]


def test_model_unloads_even_when_misinfo_judgment_fails(events, monkeypatch):
    def fail(sentences):
        raise MisinfoInferenceError("boom")

    monkeypatch.setattr(text_judge, "judge_scam", lambda sentences: SCAM)
    monkeypatch.setattr(text_judge, "run_misinfo_inference", fail)

    with pytest.raises(MisinfoInferenceError):
        text_judge.run_text_judgments(["문장"])

    assert events == ["acquire:text_judge", "unload", "release"]
