"""GPU를 쓰는 서브프로세스/HTTP 호출을 한 번에 max_concurrent개만 돌게 하는 자원 큐.
subprocess.run() 호출 하나 또는 Ollama/e5 HTTP 호출 하나 단위로 잠근다 - FastAPI 요청
전체를 잠그지 않는다(docs/superpowers/specs/2026-09-29-misinformation-detection-design.md §6).
대기열에 상한을 둬서, 너무 많이 쌓이면 타임아웃으로 조용히 죽는 대신 즉시 거절한다.
나중에 배포 트래픽 관리 작업에서 이 큐를 그대로 확장해 쓸 수 있게 대기시간을 로그로 남긴다.

작업이 끝날 때 GPU 메모리(시작 전, 작업 중 최고치)도 남긴다(2026-10-27). "모델 둘을 동시에 올리면 8GB가 모자란다"는
전제로 max_concurrent=1을 골랐지만 작업별 실측이 없었다 - 이 로그로 어떤 작업끼리 동시에 돌려도 되는지 정한다.
max_concurrent=1이면 그동안의 최고치가 곧 그 작업 몫이다. nvidia-smi가 없으면 메모리 칸 없이 기존처럼 남긴다.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Callable, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


class GpuQueueFullError(RuntimeError):
    pass


def read_gpu_memory_mb() -> Optional[int]:
    """GPU 0번의 사용 중 메모리(MB). nvidia-smi가 없거나 실패하면 None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.split()[0]) if out.returncode == 0 else None
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class _PeakSampler:
    """작업이 도는 동안 따로 돌며 GPU 메모리 최고치를 기록한다."""

    def __init__(self, read: Callable[[], Optional[int]], interval: float):
        self._read = read
        self._interval = interval
        self._stop = threading.Event()
        self.before = read()
        self.peak = self.before
        self._thread = threading.Thread(target=self._run, daemon=True) if self.before is not None else None
        if self._thread:
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            value = self._read()
            if value is not None and value > self.peak:
                self.peak = value

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()


class GpuQueue:
    def __init__(self, max_concurrent: int, max_queue_depth: int,
                 read_gpu_memory_mb: Callable[[], Optional[int]] = read_gpu_memory_mb,
                 sample_interval_seconds: float = 0.5):
        self._read_gpu_memory_mb = read_gpu_memory_mb
        self._sample_interval_seconds = sample_interval_seconds
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
        sampler = _PeakSampler(self._read_gpu_memory_mb, self._sample_interval_seconds)
        try:
            yield
        finally:
            self._semaphore.release()
            sampler.stop()
            memory = (
                f" gpu_mem_before_mb={sampler.before} gpu_mem_peak_mb={sampler.peak}"
                if sampler.before is not None else ""
            )
            logger.info(
                "gpu_queue_released label=%s run_seconds=%.3f%s",
                label,
                time.monotonic() - run_start,
                memory,
            )


@lru_cache
def get_gpu_queue() -> GpuQueue:
    settings = get_settings()
    return GpuQueue(
        max_concurrent=settings.gpu_queue_max_concurrent,
        max_queue_depth=settings.gpu_queue_max_depth,
    )
