import asyncio
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import (
    Evidence,
    MisinformationDetectionResult,
    ScamDetectionResult,
    ScamEvidence,
    VideoAnalysisResponse,
    VideoDetectionResult,
)
from app.services.dfdc_runner import DfdcInferenceError, NoFaceDetectedError, run_dfdc_inference
from app.services.misinfo_runner import MisinfoInferenceError, run_misinfo_inference
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

    error_code = None
    try:
        ai_result = await ai_task
    except NoFaceDetectedError:
        # 얼굴 없음은 정상적인 한계지 장애가 아니다(2026-09-21) - 사기감지는 얼굴 유무와
        # 무관하게 독립적으로 계산 가능하므로, 예전처럼 같이 취소하지 않고 끝까지 기다린다.
        ai_result = None
        error_code = "NO_FACE_DETECTED"
    except DfdcInferenceError as e:
        scam_task.cancel()
        raise HTTPException(status_code=502, detail=str(e)) from e

    try:
        scam_result = await scam_task
    except ScamInferenceError as e:
        # 텍스트가 원래 없어서가 아니라 파이프라인 자체가 죽은 경우다. 이걸 조용히 null로
        # 감추면 "사기 아님"으로 오인될 위험이 있어(2026-09-21), 전체 요청을 실패시킨다.
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

    misinformation_detection = None
    if scam_result.sentences:
        try:
            misinfo_result = run_misinfo_inference(scam_result.sentences)
        except MisinfoInferenceError as e:
            logger.exception("가짜정보 판정 파이프라인 실패")
            raise HTTPException(status_code=502, detail="가짜정보 판정 처리 중 오류가 발생했습니다.") from e
        misinformation_detection = MisinformationDetectionResult(
            model=misinfo_result.model,
            wiki_snapshot=misinfo_result.wiki_snapshot,
            claims=misinfo_result.claims,
        )

    ai_detection = None
    if ai_result is not None:
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
        ai_detection = VideoDetectionResult(
            model="dfdc", score=ai_result.score, evidence=evidence, evidence_image=ai_result.evidence_image
        )

    return VideoAnalysisResponse(
        ai_detection=ai_detection,
        scam_detection=scam_detection,
        misinformation_detection=misinformation_detection,
        error_code=error_code,
    )
