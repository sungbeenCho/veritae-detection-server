from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import AudioAnalysisResponse, AudioDetectionResult, Evidence
from app.services.antideepfake_runner import AntiDeepfakeInferenceError, run_antideepfake_inference

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

    try:
        result = run_antideepfake_inference(audio_bytes, file.filename or "input.wav")
    except AntiDeepfakeInferenceError as e:
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
    return AudioAnalysisResponse(
        ai_detection=AudioDetectionResult(model="antideepfake", score=result.score, evidence=evidence)
    )
