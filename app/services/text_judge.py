"""추출된 문장으로 사기 판정과 가짜정보 판정을 GPU 대기열 자리 하나 안에서 이어서 한다(2026-10-28).

두 판정은 같은 Ollama 모델을 쓴다. 사기 판정이 모델을 올리면 가짜정보 판정이 그대로 이어 쓰고, 끝나면 내린다 -
요청마다 모델을 한 번만 올린다. 자리를 판정마다 따로 잡으면 그 사이에 모델이 GPU에 남은 채 자리가 비어, 다른 요청의
Whisper(약 4.2GB)가 Ollama(약 4.1GB)와 같이 올라가 8GB를 넘을 수 있다. 그래서 자리를 놓을 때는 항상 모델을 내린다.
글자 추출(scam_runner)은 이 앞 단계에서 자기 자리를 따로 잡고, 끝나면 Whisper가 GPU에서 빠진다.
"""
import json
import logging
import urllib.request

from app.config import get_settings
from app.services.gpu_queue import get_gpu_queue
from app.services.misinfo_runner import MisinfoResult, run_misinfo_inference
from app.services.scam_judge import ScamJudgment, judge_scam

logger = logging.getLogger(__name__)


class TextJudgeResult:
    def __init__(self, scam: ScamJudgment, misinfo: MisinfoResult):
        self.scam = scam
        self.misinfo = misinfo


def unload_ollama_model() -> None:
    """실패해도 요청은 실패시키지 않는다 - 로그만 남긴다(misinfo_infer.unload_ollama_model과 같은 방식).
    가짜정보 판정이 정상으로 끝났으면 이미 내려가 있어 바로 끝난다."""
    settings = get_settings()
    req = urllib.request.Request(
        f"{settings.ollama_url}/api/generate",
        data=json.dumps({"model": settings.ollama_model, "keep_alive": 0}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=30).close()
    except Exception:
        logger.exception("Ollama 모델 내리기 실패 - 직접 `ollama stop %s` 필요", settings.ollama_model)


def run_text_judgments(sentences: list[str]) -> TextJudgeResult:
    """ScamJudgeError/MisinfoInferenceError는 그대로 올린다 - 라우터가 어느 판정이 실패했는지 구분해 응답한다."""
    with get_gpu_queue().acquire("text_judge"):
        try:
            scam = judge_scam(sentences)
            misinfo = run_misinfo_inference(sentences)
        finally:
            unload_ollama_model()
    return TextJudgeResult(scam=scam, misinfo=misinfo)
