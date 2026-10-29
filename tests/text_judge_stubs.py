"""라우터 테스트 공용 - 사기 판정(Ollama)·모델 내리기·GPU 대기열을 실제로 부르지 않게 바꿔 끼운다.

가짜정보 쪽은 각 테스트가 text_judge.run_misinfo_inference를 바꿔 끼운다. 라우터 테스트 파일에서
`from tests.text_judge_stubs import FAKE_SCAM, stub_text_judge  # noqa: F401`로 가져오면 autouse로 적용된다.
"""
from contextlib import contextmanager

import pytest

from app.services import text_judge
from app.services.scam_judge import ScamJudgment

FAKE_SCAM = ScamJudgment(model="qwen3.5:4b-scam-v3", score=0.82, reason="은행을 사칭해 계좌번호를 요구하는 사기 유형이다.")
FAKE_SCAM_JSON = {
    "model": "qwen3.5:4b-scam-v3",
    "score": 0.82,
    "evidence": [{"sentence": "은행을 사칭해 계좌번호를 요구하는 사기 유형이다.", "score": 0.82}],
}


class _NoopGpuQueue:
    @contextmanager
    def acquire(self, label):
        yield


@pytest.fixture(autouse=True)
def stub_text_judge(monkeypatch):
    monkeypatch.setattr(text_judge, "judge_scam", lambda sentences: FAKE_SCAM)
    monkeypatch.setattr(text_judge, "unload_ollama_model", lambda: None)
    monkeypatch.setattr(text_judge, "get_gpu_queue", lambda: _NoopGpuQueue())
