"""GPU를 쓰는 서브프로세스/HTTP 호출을 한 번에 max_concurrent개만 돌게 하는 자원 큐.
subprocess.run() 호출 하나 또는 Ollama/e5 HTTP 호출 하나 단위로 잠근다 - FastAPI 요청
전체를 잠그지 않는다(docs/superpowers/specs/2026-09-29-misinformation-detection-design.md §6).
대기열에 상한을 둬서, 너무 많이 쌓이면 타임아웃으로 조용히 죽는 대신 즉시 거절한다.
나중에 배포 트래픽 관리 작업에서 이 큐를 그대로 확장해 쓸 수 있게 대기시간을 로그로 남긴다.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)


class GpuQueueFullError(RuntimeError):
    pass


class GpuQueue:
    def __init__(self, max_concurrent: int, max_queue_depth: int):
        self._semaphore = threading.Semaphore(max_concurrent)
        self._max_queue_depth = max_queue_depth
        self._lock = threading.Lock()
        self._waiting = 0

    @contextmanager
    def acquire(self, label: str):
        with self._lock:
            if self._waiting >= self._max_queue_depth:
                logger.warning("gpu_queue_full label=%s waiting=%d", label, self._waiting)
                raise GpuQueueFullError(
                    f"GPU 자원 큐가 가득 찼습니다(대기 {self._max_queue_depth}건 초과)"
                )
            self._waiting += 1
        wait_start = time.monotonic()
        try:
            self._semaphore.acquire()
        finally:
            with self._lock:
                self._waiting -= 1
        wait_seconds = time.monotonic() - wait_start
        logger.info("gpu_queue_acquired label=%s wait_seconds=%.3f", label, wait_seconds)
        run_start = time.monotonic()
        try:
            yield
        finally:
            self._semaphore.release()
            logger.info(
                "gpu_queue_released label=%s run_seconds=%.3f",
                label,
                time.monotonic() - run_start,
            )


@lru_cache
def get_gpu_queue() -> GpuQueue:
    settings = get_settings()
    return GpuQueue(
        max_concurrent=settings.gpu_queue_max_concurrent,
        max_queue_depth=settings.gpu_queue_max_depth,
    )
