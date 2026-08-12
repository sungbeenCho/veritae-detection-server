from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import AIDetectionResult, ImageAnalysisResponse
from app.services.spai_runner import SpaiInferenceError, run_spai_inference

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/process/image", response_model=ImageAnalysisResponse)
async def process_image(file: UploadFile = File(...)) -> ImageAnalysisResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415, detail=f"unsupported content type: {file.content_type}"
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        score = run_spai_inference(image_bytes, file.filename or "input.jpg")
    except SpaiInferenceError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return ImageAnalysisResponse(
        ai_detection=AIDetectionResult(model="spai", score=score)
    )
