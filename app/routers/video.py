import asyncio
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import Evidence, ScamDetectionResult, ScamEvidence, VideoAnalysisResponse, VideoDetectionResult
from app.services.dfdc_runner import DfdcInferenceError, NoFaceDetectedError, run_dfdc_inference
from app.services.scam_runner import ScamInferenceError, ScamResult, run_scam_inference_video

logger = logging.getLogger(__name__)

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

    filename = file.filename or "input.mp4"
    ai_task = asyncio.create_task(asyncio.to_thread(run_dfdc_inference, video_bytes, filename))
    scam_task = asyncio.create_task(asyncio.to_thread(run_scam_inference_video, video_bytes, filename))

    try:
        ai_result = await ai_task
    except NoFaceDetectedError as e:
        scam_task.cancel()
        raise HTTPException(status_code=422, detail=str(e)) from e
    except DfdcInferenceError as e:
        scam_task.cancel()
        raise HTTPException(status_code=502, detail=str(e)) from e

    try:
        scam_result = await scam_task
    except ScamInferenceError:
        logger.exception("사기감지 파이프라인 실패 (best-effort, 요청은 계속 진행)")
        scam_result = ScamResult(None, [])

    evidence = [
        Evidence(
            title=e["title"],
            description=e["description"],
            tags=e["tags"],
            start_sec=e["start_sec"],
            end_sec=e["end_sec"],
        )
        for e in ai_result.evidence
    ]
    scam_detection = (
        ScamDetectionResult(
            model="lilju",
            score=scam_result.score,
            evidence=[ScamEvidence(**e) for e in scam_result.evidence],
        )
        if scam_result.score is not None
        else None
    )

    return VideoAnalysisResponse(
        ai_detection=VideoDetectionResult(
            model="dfdc", score=ai_result.score, evidence=evidence, evidence_image=ai_result.evidence_image
        ),
        scam_detection=scam_detection,
    )
