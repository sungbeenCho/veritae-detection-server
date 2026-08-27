import json
import shutil
import subprocess
import uuid
from pathlib import Path, PureWindowsPath

from app.config import get_settings


class DfdcInferenceError(RuntimeError):
    pass


class NoFaceDetectedError(DfdcInferenceError):
    """dfdc_infer.py가 종료 코드 2(얼굴 없음 전용)로 끝난 경우. 일반 DfdcInferenceError와
    구분해서 video.py가 502가 아니라 422로 응답할 수 있게 한다(2026-08-27)."""


class DfdcResult:
    def __init__(self, score: float, evidence: list[dict], evidence_image: str | None):
        self.score = score
        self.evidence = evidence
        self.evidence_image = evidence_image


def _safe_filename(filename: str) -> str:
    # PureWindowsPath treats both / and \ as separators, so this strips any
    # directory components regardless of host OS, preventing a crafted upload
    # filename from writing outside the job's temp dir. (spai_runner.py/antideepfake_runner.py와 동일 로직)
    name = PureWindowsPath(filename).name
    return name if name and name not in (".", "..") else "upload"


def run_dfdc_inference(video_bytes: bytes, filename: str) -> DfdcResult:
    settings = get_settings()
    job_dir = settings.dfdc_work_dir / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    input_file = job_dir / _safe_filename(filename)
    output_file = job_dir / "result.json"
    input_file.write_bytes(video_bytes)

    command = [
        settings.dfdc_python,
        str(settings.dfdc_script),
        "--repo-dir", str(settings.dfdc_repo_dir),
        "--checkpoints", *(str(c) for c in settings.dfdc_checkpoints),
        "--input", str(input_file),
        "--output", str(output_file),
    ]

    try:
        result = subprocess.run(
            command,
            cwd=settings.dfdc_repo_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=settings.dfdc_timeout_seconds,
        )

        if result.returncode == 2:
            raise NoFaceDetectedError(result.stderr[-2000:])

        if result.returncode != 0:
            raise DfdcInferenceError(f"DFDC inference failed: {result.stderr[-2000:]}")

        if not output_file.exists():
            raise DfdcInferenceError(f"expected output JSON not found: {output_file}")

        return _parse_result(output_file)
    except subprocess.TimeoutExpired as e:
        raise DfdcInferenceError(
            f"DFDC inference timed out after {settings.dfdc_timeout_seconds}s"
        ) from e
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def _parse_result(output_file: Path) -> DfdcResult:
    try:
        data = json.loads(output_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        raise DfdcInferenceError(f"DFDC output JSON을 읽거나 파싱할 수 없습니다: {output_file}") from e
    if "score" not in data:
        raise DfdcInferenceError(f"DFDC output JSON missing 'score' field: {output_file}")
    return DfdcResult(
        score=data["score"],
        evidence=data.get("evidence", []),
        evidence_image=data.get("evidence_image"),
    )
