from pydantic import BaseModel


class AIDetectionResult(BaseModel):
    model: str
    score: float


class ImageAnalysisResponse(BaseModel):
    ai_detection: AIDetectionResult
