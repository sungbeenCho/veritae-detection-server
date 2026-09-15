import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path, PureWindowsPath

from app.config import get_settings

# EasyOCR이 모델을 처음 다운로드할 때 진행률 표시줄에 유니코드 블록 문자(█)를 print하는데,
# Windows 콘솔 기본 코드페이지(cp949)로는 이 문자를 인코딩할 수 없어 자식 프로세스가
# UnicodeEncodeError로 죽는다(3060Ti 실기 확인, 2026-09-15). 자식 파이썬 프로세스의
# stdout/stderr 인코딩을 UTF-8로 강제해 방지한다.
_SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


class ScamInferenceError(RuntimeError):
    pass


class ScamResult:
    def __init__(self, score: float | None, evidence: list[dict]):
        self.score = score
        self.evidence = evidence


def _safe_filename(filename: str) -> str:
    # PureWindowsPath treats both / and \ as separators, so this strips any
    # directory components regardless of host OS (spai_runner.py 등과 동일 로직).
    name = PureWindowsPath(filename).name
    return name if name and name not in (".", "..") else "upload"


def _run_scam_infer(mode: str, input_bytes: bytes, filename: str) -> ScamResult:
    settings = get_settings()
    job_dir = settings.text_extraction_work_dir / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    input_file = job_dir / _safe_filename(filename)
    output_file = job_dir / "result.json"
    input_file.write_bytes(input_bytes)

    command = [
        settings.text_extraction_python,
        str(settings.text_extraction_script),
        "--mode", mode,
        "--input", str(input_file),
        "--output", str(output_file),
        "--lilju-model-id", settings.lilju_model_id,
        "--paddleocr-lang", settings.paddleocr_lang,
        "--whisper-model-size", settings.whisper_model_size,
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=settings.text_extraction_timeout_seconds,
            env=_SUBPROCESS_ENV,
        )

        if result.returncode != 0:
            raise ScamInferenceError(f"사기감지 추론 실패: {result.stderr[-2000:]}")

        if not output_file.exists():
            raise ScamInferenceError(f"expected output JSON not found: {output_file}")

        return _parse_result(output_file)
    except subprocess.TimeoutExpired as e:
        raise ScamInferenceError(
            f"사기감지 추론이 {settings.text_extraction_timeout_seconds}초 안에 끝나지 않았습니다"
        ) from e
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def _parse_result(output_file: Path) -> ScamResult:
    try:
        data = json.loads(output_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        raise ScamInferenceError(f"사기감지 output JSON을 읽거나 파싱할 수 없습니다: {output_file}") from e
    return ScamResult(score=data.get("score"), evidence=data.get("evidence", []))


def run_scam_inference_image(image_bytes: bytes, filename: str) -> ScamResult:
    return _run_scam_infer("ocr", image_bytes, filename)


def run_scam_inference_audio(audio_bytes: bytes, filename: str) -> ScamResult:
    return _run_scam_infer("stt", audio_bytes, filename)


def run_scam_inference_video(video_bytes: bytes, filename: str) -> ScamResult:
    """faster-whisper는 오디오 입력만 받으므로, 영상에서 오디오 트랙만 ffmpeg로 뽑아낸
    뒤 STT 모드로 넘긴다(antideepfake_infer.py의 압축포맷→wav 변환과 동일한 ffmpeg
    서브프로세스 패턴)."""
    settings = get_settings()
    job_dir = settings.text_extraction_work_dir / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    video_file = job_dir / _safe_filename(filename)
    audio_file = job_dir / "audio.wav"
    video_file.write_bytes(video_bytes)

    try:
        ffmpeg_result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_file), "-vn", "-acodec", "pcm_s16le", str(audio_file)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=settings.text_extraction_timeout_seconds,
        )
        if ffmpeg_result.returncode != 0:
            raise ScamInferenceError(f"영상에서 오디오 트랙 추출 실패: {ffmpeg_result.stderr[-2000:]}")

        return _run_scam_infer("stt", audio_file.read_bytes(), "audio.wav")
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
