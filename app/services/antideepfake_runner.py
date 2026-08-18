import json
import shutil
import subprocess
import uuid
from pathlib import Path, PureWindowsPath

from app.config import get_settings


class AntiDeepfakeInferenceError(RuntimeError):
    pass


class AntiDeepfakeResult:
    def __init__(self, score: float, evidence: list[dict]):
        self.score = score
        self.evidence = evidence


def _safe_filename(filename: str) -> str:
    # PureWindowsPath treats both / and \ as separators, so this strips any
    # directory components regardless of host OS, preventing a crafted upload
    # filename from writing outside the job's temp dir. (spai_runner.py와 동일 로직)
    name = PureWindowsPath(filename).name
    return name if name and name not in (".", "..") else "upload"


def run_antideepfake_inference(audio_bytes: bytes, filename: str) -> AntiDeepfakeResult:
    settings = get_settings()
    job_dir = settings.antideepfake_work_dir / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    input_file = job_dir / _safe_filename(filename)
    output_file = job_dir / "result.json"
    input_file.write_bytes(audio_bytes)

    command = [
        settings.antideepfake_python,
        str(settings.antideepfake_script),
        "--repo-dir", str(settings.antideepfake_repo_dir),
        "--checkpoint", str(settings.antideepfake_checkpoint),
        "--input", str(input_file),
        "--output", str(output_file),
    ]

    try:
        result = subprocess.run(
            command,
            cwd=settings.antideepfake_repo_dir,
            capture_output=True,
            text=True,
            timeout=settings.antideepfake_timeout_seconds,
        )

        if result.returncode != 0:
            raise AntiDeepfakeInferenceError(f"AntiDeepfake inference failed: {result.stderr[-2000:]}")

        if not output_file.exists():
            raise AntiDeepfakeInferenceError(f"expected output JSON not found: {output_file}")

        return _parse_result(output_file)
    except subprocess.TimeoutExpired as e:
        raise AntiDeepfakeInferenceError(
            f"AntiDeepfake inference timed out after {settings.antideepfake_timeout_seconds}s"
        ) from e
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def _parse_result(output_file: Path) -> AntiDeepfakeResult:
    try:
        data = json.loads(output_file.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        raise AntiDeepfakeInferenceError(f"AntiDeepfake output JSON을 읽거나 파싱할 수 없습니다: {output_file}") from e
    if "score" not in data:
        raise AntiDeepfakeInferenceError(f"AntiDeepfake output JSON missing 'score' field: {output_file}")
    return AntiDeepfakeResult(score=data["score"], evidence=data.get("evidence", []))
