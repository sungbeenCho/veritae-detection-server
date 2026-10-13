import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.logging_config import configure_logging
from app.routers import audio, image, video
from app.services.gpu_queue import GpuQueueFullError

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Veritae Detection Server")

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


@app.exception_handler(GpuQueueFullError)
async def gpu_queue_full_handler(request: Request, exc: GpuQueueFullError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
