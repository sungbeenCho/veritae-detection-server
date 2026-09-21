import asyncio
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import AIDetectionResult, ImageAnalysisResponse, ScamDetectionResult, ScamEvidence
from app.services.scam_runner import ScamInferenceError, run_scam_inference_image
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
    except ScamInferenceError as e:
        # 텍스트가 원래 없어서가 아니라 파이프라인 자체가 죽은 경우다. 이걸 조용히 null로
        # 감추면 "사기 아님"으로 오인될 위험이 있어(video.py와 동일 원칙, 2026-09-21 변경 -
        # 예전엔 best-effort로 삼켰었으나 이 안전 문제 때문에 정책을 바꿈), 전체 요청을 실패시킨다.
        # 클라이언트에는 일반화된 메시지만 주고, 실제 원인은 서버 로그에만 남긴다.
        logger.exception("사기감지 파이프라인 실패")
        raise HTTPException(status_code=502, detail="사기감지 처리 중 오류가 발생했습니다.") from e

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
