from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routers import audio, image, video
from app.services.gpu_queue import GpuQueueFullError

app = FastAPI(title="Veritae Detection Server")

app.include_router(image.router)
app.include_router(audio.router)
app.include_router(video.router)


@app.exception_handler(GpuQueueFullError)
async def gpu_queue_full_handler(request: Request, exc: GpuQueueFullError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
