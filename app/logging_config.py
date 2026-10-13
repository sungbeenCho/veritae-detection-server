"""앱 로그를 서버 콘솔에 보이게 한다.

uvicorn은 자기 로거(uvicorn.*)만 설정한다. 루트 로거에는 핸들러가 없어서 앱 코드의 logger.info(GPU 대기열의
gpu_queue_acquired/released 시간 등)는 어디에도 출력되지 않고 WARNING 이상만 보였다(2026-10-13 확인) -
그래서 어느 단계가 느린지, 판정이 왜 비었는지 서버 로그로 알 수 없었다.
"""
from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging() -> None:
    """"app" 로거에 INFO 이상을 stderr로 내보내는 핸들러를 하나만 붙인다(여러 번 불러도 중복되지 않는다)."""
    app_logger = logging.getLogger("app")
    if app_logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    app_logger.addHandler(handler)
    app_logger.setLevel(logging.INFO)
