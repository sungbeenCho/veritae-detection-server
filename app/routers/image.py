import asyncio
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import AIDetectionResult, ImageAnalysisResponse, ScamDetectionResult, ScamEvidence
from app.services.scam_runner import ScamInferenceError, ScamResult, run_scam_inference_image
from app.services.spai_runner import SpaiInferenceError, run_spai_inference

logger = logging.getLogger(__name__)

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

    filename = file.filename or "input.jpg"
    ai_task = asyncio.create_task(asyncio.to_thread(run_spai_inference, image_bytes, filename))
    scam_task = asyncio.create_task(asyncio.to_thread(run_scam_inference_image, image_bytes, filename))

    try:
        ai_result = await ai_task
    except SpaiInferenceError as e:
        scam_task.cancel()
        raise HTTPException(status_code=502, detail=str(e)) from e

    try:
        scam_result = await scam_task
    except ScamInferenceError:
        # 사기감지는 best-effort - AI판독(SPAI)이 성공했으면 사기감지 실패로 전체 요청을
        # 실패시키지 않는다(evidenceImage/Grad-CAM과 동일 원칙, 2026-09-13 설계 확정).
        logger.exception("사기감지 파이프라인 실패 (best-effort, 요청은 계속 진행)")
        scam_result = ScamResult(None, [])

    scam_detection = (
        ScamDetectionResult(
            model="lilju",
            score=scam_result.score,
            evidence=[ScamEvidence(**e) for e in scam_result.evidence],
        )
        if scam_result.score is not None
        else None
    )

    return ImageAnalysisResponse(
        ai_detection=AIDetectionResult(
            model="spai", score=ai_result.score, evidence_image=ai_result.evidence_image
        ),
        scam_detection=scam_detection,
    )
