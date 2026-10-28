import hmac
import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.logging_config import configure_logging
from app.routers import audio, image, video
from app.services.gpu_queue import GpuQueueFullError

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Veritae Detection Server")

# Spring 서버만 분석을 시킬 수 있게 하는 공유 비밀 키(2026-10-28, 배포 준비). 이 서버는 0.0.0.0으로 열려 있어
# 집 공유기 안·Tailscale 안의 다른 기기도 닿을 수 있다 - 키가 맞는 요청만 GPU를 쓰게 한다.
# 비워 두면(로컬 개발) 검사하지 않는다. 데스크탑 운영 환경에서는 반드시 설정한다(README).
API_KEY = os.environ.get("DETECTION_API_KEY", "")
API_KEY_HEADER = "X-Detection-Key"
if not API_KEY:
    logger.warning("DETECTION_API_KEY가 비어 있어 키 검사 없이 모든 분석 요청을 받습니다(로컬 개발 전용).")

app.include_router(image.router)
app.include_router(audio.router)
app.include_router(video.router)


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    """요청 하나가 총 몇 초 걸렸는지 남긴다(상태 코드 포함 - 502/503으로 끝난 요청도 보이게)."""
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        if request.url.path != "/health":
            logger.info(
                "request_done method=%s path=%s status=%d seconds=%.1f",
                request.method, request.url.path, status, time.perf_counter() - start,
            )


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if API_KEY and request.url.path != "/health":
        given = request.headers.get(API_KEY_HEADER, "")
        if not hmac.compare_digest(given.encode(), API_KEY.encode()):
            logger.warning("api_key_rejected path=%s client=%s", request.url.path, request.client.host if request.client else "-")
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return await call_next(request)


@app.exception_handler(GpuQueueFullError)
async def gpu_queue_full_handler(request: Request, exc: GpuQueueFullError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
