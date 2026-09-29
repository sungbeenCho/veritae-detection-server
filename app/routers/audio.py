import asyncio
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import (
    AudioAnalysisResponse,
    AudioDetectionResult,
    Evidence,
    MisinformationDetectionResult,
    ScamDetectionResult,
    ScamEvidence,
)
from app.services.antideepfake_runner import AntiDeepfakeInferenceError, run_antideepfake_inference
from app.services.media_probe import probe_audio_duration_seconds
from app.services.misinfo_runner import MisinfoInferenceError, run_misinfo_inference
from app.services.scam_runner import ScamInferenceError, run_scam_inference_audio

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/aac"}
MAX_AUDIO_DURATION_SECONDS = 300.0


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

    duration_seconds = await asyncio.to_thread(probe_audio_duration_seconds, audio_bytes, filename)
    if duration_seconds is not None and duration_seconds > MAX_AUDIO_DURATION_SECONDS:
        raise HTTPException(
            status_code=400,
            detail={"code": "AUDIO_TOO_LONG", "message": "음성 길이가 5분을 초과합니다."},
        )

    ai_task = asyncio.create_task(asyncio.to_thread(run_antideepfake_inference, audio_bytes, filename))
    scam_task = asyncio.create_task(asyncio.to_thread(run_scam_inference_audio, audio_bytes, filename))

    try:
        ai_result = await ai_task
    except AntiDeepfakeInferenceError as e:
        scam_task.cancel()
        raise HTTPException(status_code=502, detail=str(e)) from e

    try:
        scam_result = await scam_task
    except ScamInferenceError as e:
        # 텍스트가 원래 없어서가 아니라 파이프라인 자체가 죽은 경우다. 이걸 조용히 null로
        # 감추면 "사기 아님"으로 오인될 위험이 있어(image.py와 동일 원칙, 2026-09-21 변경 -
        # 예전엔 best-effort로 삼켰었으나 이 안전 문제 때문에 정책을 바꿈), 전체 요청을 실패시킨다.
        # 클라이언트에는 일반화된 메시지만 주고, 실제 원인은 서버 로그에만 남긴다.
        logger.exception("사기감지 파이프라인 실패")
        raise HTTPException(status_code=502, detail="사기감지 처리 중 오류가 발생했습니다.") from e

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

    return AudioAnalysisResponse(
        ai_detection=AudioDetectionResult(model="antideepfake", score=ai_result.score, evidence=evidence),
        scam_detection=scam_detection,
        misinformation_detection=misinformation_detection,
    )
