from pydantic import BaseModel


class AIDetectionResult(BaseModel):
    model: str
    score: float


class ImageAnalysisResponse(BaseModel):
    ai_detection: AIDetectionResult


class Evidence(BaseModel):
    title: str
    description: str
    tags: list[str]
    start_sec: float
    end_sec: float


class AudioDetectionResult(BaseModel):
    model: str
    score: float
    evidence: list[Evidence]


class AudioAnalysisResponse(BaseModel):
    ai_detection: AudioDetectionResult


class VideoDetectionResult(BaseModel):
    model: str
    score: float
    evidence: list[Evidence]
    evidence_image: str | None = None


class VideoAnalysisResponse(BaseModel):
    ai_detection: VideoDetectionResult
