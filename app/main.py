from fastapi import FastAPI

from app.routers import audio, image

app = FastAPI(title="Veritae Detection Server")

app.include_router(image.router)
app.include_router(audio.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
