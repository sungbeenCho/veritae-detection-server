import asyncio
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import AudioAnalysisResponse, AudioDetectionResult, Evidence, ScamDetectionResult, ScamEvidence
from app.services.antideepfake_runner import AntiDeepfakeInferenceError, run_antideepfake_inference
from app.services.scam_runner import ScamInferenceError, ScamResult, run_scam_inference_audio

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/aac"}


@router.post("/process/audio", response_model=AudioAnalysisResponse)
async def process_audio(file: UploadFile = File(...)) -> AudioAnalysisResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415, detail=f"unsupported content type: {file.content_type}"
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    filename = file.filename or "input.wav"
    ai_task = asyncio.create_task(asyncio.to_thread(run_antideepfake_inference, audio_bytes, filename))
    scam_task = asyncio.create_task(asyncio.to_thread(run_scam_inference_audio, audio_bytes, filename))

    try:
        ai_result = await ai_task
    except AntiDeepfakeInferenceError as e:
        scam_task.cancel()
        raise HTTPException(status_code=502, detail=str(e)) from e

    try:
        scam_result = await scam_task
    except ScamInferenceError:
        # 사기감지는 best-effort - AI판독(AntiDeepfake)이 성공했으면 사기감지 실패로
        # 전체 요청을 실패시키지 않는다(image.py와 동일 원칙, 2026-09-13 설계 확정).
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

    return AudioAnalysisResponse(
        ai_detection=AudioDetectionResult(model="antideepfake", score=ai_result.score, evidence=evidence),
        scam_detection=scam_detection,
    )
