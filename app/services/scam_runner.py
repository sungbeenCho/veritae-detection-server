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


def _run_scam_infer_video(job_dir: Path, audio_file: Path, frames_dir: Path | None) -> ScamResult:
    """`_run_scam_infer`와 달리 자기 자신의 job_dir을 새로 만들지 않는다 - 호출자
    (run_scam_inference_video)가 이미 만들어둔 job_dir 안에 audio_file/frames_dir이
    준비되어 있다고 가정하고, 그 경로들을 그대로 scam_infer.py --mode video에 넘긴다
    (2026-09-15 추가, 영상 화면 텍스트 인식용)."""
    settings = get_settings()
    output_file = job_dir / "result.json"

    command = [
        settings.text_extraction_python,
        str(settings.text_extraction_script),
        "--mode", "video",
        "--input", str(audio_file),
        "--output", str(output_file),
        "--lilju-model-id", settings.lilju_model_id,
        "--paddleocr-lang", settings.paddleocr_lang,
        "--whisper-model-size", settings.whisper_model_size,
    ]
    if frames_dir is not None:
        command += ["--frames-dir", str(frames_dir)]

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
    서브프로세스 패턴). 이 오디오 추출 부분은 기존 그대로 - 아래 프레임 추출은 화면
    텍스트(자막/문구)도 같이 보기 위해 2026-09-15 추가된 별도 경로다."""
    settings = get_settings()
    job_dir = settings.text_extraction_work_dir / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    video_file = job_dir / _safe_filename(filename)
    audio_file = job_dir / "audio.wav"
    frames_dir = job_dir / "frames"
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

        # 화면 텍스트 인식용 프레임 추출(3초에 한 장, 보수적 샘플링 - 실측 근거 없는
        # 초기값, 2026-09-15). 실패해도 전체를 실패시키지 않는다 - 음성만으로도
        # best-effort 진행 가능(기존 사기감지 best-effort 원칙과 동일).
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_extract_result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_file), "-vf", "fps=1/3", str(frames_dir / "frame_%04d.png")],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=settings.text_extraction_timeout_seconds,
        )
        frames_dir_arg = frames_dir if frame_extract_result.returncode == 0 else None

        return _run_scam_infer_video(job_dir, audio_file, frames_dir_arg)
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
