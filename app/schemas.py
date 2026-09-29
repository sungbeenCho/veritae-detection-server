from pydantic import BaseModel


class ScamEvidence(BaseModel):
    sentence: str
    score: float


class ScamDetectionResult(BaseModel):
    model: str
    score: float
    evidence: list[ScamEvidence]


class WikiEvidence(BaseModel):
    title: str
    text: str
    url: str


class MisinformationClaim(BaseModel):
    sentence: str
    reason: str
    evidence: list[WikiEvidence]


class MisinformationDetectionResult(BaseModel):
    model: str
    wiki_snapshot: str
    claims: list[MisinformationClaim]


class AIDetectionResult(BaseModel):
    model: str
    score: float
    evidence_image: str | None = None


class ImageAnalysisResponse(BaseModel):
    ai_detection: AIDetectionResult
    scam_detection: ScamDetectionResult | None = None
    misinformation_detection: MisinformationDetectionResult | None = None


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
    scam_detection: ScamDetectionResult | None = None
    misinformation_detection: MisinformationDetectionResult | None = None


class VideoDetectionResult(BaseModel):
    model: str
    score: float
    evidence: list[Evidence]
    evidence_image: str | None = None


class VideoAnalysisResponse(BaseModel):
    # ai_detection은 얼굴을 못 찾으면 null일 수 있다(정상적인 한계, 에러 아님) - 이때
    # error_code가 이유를 알려준다. ai_detection이 있으면 error_code는 항상 null이다.
    ai_detection: VideoDetectionResult | None
    scam_detection: ScamDetectionResult | None = None
    misinformation_detection: MisinformationDetectionResult | None = None
    error_code: str | None = None
