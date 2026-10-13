"""misinfo_infer.py가 단계별 시간을 로그(stderr)로 남기는 데 쓰는 작은 도구들. 실제 모델 없이 테스트한다.

서버(misinfo_runner.py)가 이 하위 프로세스의 stderr를 서버 로그로 올린다. 이미지 한 장(문장 2개)이 2분 걸렸는데
어느 단계 때문인지 알 방법이 없었다(2026-10-13) - 문장마다 단계별 시간과 "왜 결과가 비었는지"를 남긴다.
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

logger = logging.getLogger("misinfo")


class StepTimer:
    """문장 하나를 처리하는 동안 단계별(검색, 재정렬, LLM, NLI...) 시간을 모으고, 결과(outcome)를 기록한다."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.outcome = "unknown"

    @contextmanager
    def step(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start

    def summary(self) -> str:
        return " ".join(f"{name}={seconds:.1f}s" for name, seconds in self.seconds.items())


def format_ollama_stats(data: dict, wall_seconds: float) -> str:
    """Ollama 응답이 알려주는 시간(나노초): 모델 로드, 근거(프롬프트) 읽기, 생성."""
    ns = 1_000_000_000
    return (
        f"wall={wall_seconds:.1f}s load={data.get('load_duration', 0) / ns:.1f}s "
        f"prompt_eval={data.get('prompt_eval_duration', 0) / ns:.1f}s({data.get('prompt_eval_count', 0)}tok) "
        f"eval={data.get('eval_duration', 0) / ns:.1f}s({data.get('eval_count', 0)}tok)"
    )


@contextmanager
def logged_step(name: str):
    """모델 로드처럼 문장과 무관한 일회성 단계의 시간을 `name seconds=...`로 남긴다."""
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s seconds=%.1f", name, time.perf_counter() - start)


def configure_script_logging() -> None:
    """하위 프로세스의 INFO 이상 로그를 stderr로 내보낸다(서버가 모아서 서버 로그에 올린다). 중복 설치하지 않는다."""
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
