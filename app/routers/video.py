from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import Evidence, VideoAnalysisResponse, VideoDetectionResult
from app.services.dfdc_runner import DfdcInferenceError, run_dfdc_inference

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo"}


@router.post("/process/video", response_model=VideoAnalysisResponse)
async def process_video(file: UploadFile = File(...)) -> VideoAnalysisResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415, detail=f"unsupported content type: {file.content_type}"
        )

    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        result = run_dfdc_inference(video_bytes, file.filename or "input.mp4")
    except DfdcInferenceError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    evidence = [
        Evidence(
            title=e["title"],
            description=e["description"],
            tags=e["tags"],
            start_sec=e["start_sec"],
            end_sec=e["end_sec"],
        )
        for e in result.evidence
    ]
    return VideoAnalysisResponse(
        ai_detection=VideoDetectionResult(
            model="dfdc", score=result.score, evidence=evidence, evidence_image=result.evidence_image
        )
    )
