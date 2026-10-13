"""처리 단계(SPAI, 글자 추출+사기감지, 가짜정보 판정 등)가 각각 몇 초 걸리는지 서버 로그에 남긴다.
이미지 한 장이 2분 걸렸는데 어느 단계 때문인지 알 방법이 없었다(2026-10-13)."""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def timed(label: str, fn: Callable[..., T]) -> Callable[..., T]:
    """fn을 감싸 실행 시간을 `stage_done label=... seconds=...`로 남긴다. 실패하면 `stage_failed`로 남기고 예외는 그대로 던진다.
    fn은 호출 시점에 넘겨받는다 - 라우터가 모듈 속성을 그때그때 읽어 오므로 테스트에서 바꿔 끼운 함수도 그대로 동작한다."""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except BaseException:
            logger.info("stage_failed label=%s seconds=%.1f", label, time.perf_counter() - start)
            raise
        logger.info("stage_done label=%s seconds=%.1f", label, time.perf_counter() - start)
        return result

    return wrapper
