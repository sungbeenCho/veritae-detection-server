import base64
import csv
import shutil
import subprocess
import uuid
from pathlib import Path, PureWindowsPath

from app.config import get_settings

SPAI_SCORE_TAG = "spai"
# antideepfake_infer.py/dfdc_infer.py와 동일한 임계값 - score가 이 미만이면(진짜 사진일
# 가능성이 높으면) 근거 히트맵 자체를 만들지 않는다(2026-09-12, 사용자 확인).
EVIDENCE_SCORE_THRESHOLD = 0.5


class SpaiInferenceError(RuntimeError):
    pass


class SpaiResult:
    def __init__(self, score: float, evidence_image: str | None):
        self.score = score
        self.evidence_image = evidence_image


def _safe_filename(filename: str) -> str:
    # PureWindowsPath treats both / and \ as separators, so this strips any
    # directory components regardless of host OS, preventing a crafted upload
    # filename from writing outside the job's temp dir.
    name = PureWindowsPath(filename).name
    return name if name and name not in (".", "..") else "upload"


def _find_and_encode_overlay(output_dir: Path) -> str | None:
    """TEST.EXPORT_IMAGE_PATCHES 로 SPAI가 내보낸 판독 근거 히트맵(원본 사진 위에 합성된
    오버레이, attn_overlay_<score>.png)을 찾아 base64로 인코딩한다. best-effort - 파일이
    없거나 읽기 실패해도 None을 반환하고 전체 분석은 계속 성공한다(dfdc_infer.py의
    try_generate_heatmap과 동일 원칙). 실제 경로는
    <output_dir>/images/<dataset_idx>/patches_attn/attn_overlay_<score>.png 이고 score가
    파일명에 붙어 예측 불가능해 glob으로 찾는다(mever-team/spai의 models/sid.py 소스로
    2026-09-12 검증 완료).
    """
    try:
        matches = sorted(output_dir.glob("**/patches_attn/attn_overlay_*.png"))
        if not matches:
            return None
        return base64.b64encode(matches[0].read_bytes()).decode("ascii")
    except OSError:
        return None


def run_spai_inference(image_bytes: bytes, filename: str) -> SpaiResult:
    settings = get_settings()
    job_dir = settings.work_dir / uuid.uuid4().hex
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    (input_dir / _safe_filename(filename)).write_bytes(image_bytes)

    command = [
        settings.spai_python,
        "-m", "spai", "infer",
        "--cfg", settings.spai_cfg,
        "--model", settings.spai_model,
        "--input", str(input_dir),
        "--output", str(output_dir),
        "--tag", SPAI_SCORE_TAG,
        "--resize-to", str(settings.spai_resize_to),
        "--opt", "TEST.EXPORT_IMAGE_PATCHES", "True",
    ]

    try:
        result = subprocess.run(
            command,
            cwd=settings.spai_repo_dir,
            capture_output=True,
            text=True,
            timeout=settings.spai_timeout_seconds,
        )

        if result.returncode != 0:
            raise SpaiInferenceError(f"SPAI inference failed: {result.stderr[-2000:]}")

        # SPAI names the output CSV after the input folder ("input.csv" here)
        # and writes the score into a column named after --tag.
        output_csv = output_dir / "input.csv"
        if not output_csv.exists():
            raise SpaiInferenceError(f"expected output CSV not found: {output_csv}")

        with output_csv.open(newline="") as f:
            row = next(csv.DictReader(f), None)

        if row is None or SPAI_SCORE_TAG not in row:
            raise SpaiInferenceError(
                f"SPAI output CSV missing expected '{SPAI_SCORE_TAG}' score column"
            )

        score = float(row[SPAI_SCORE_TAG])
        evidence_image = (
            _find_and_encode_overlay(output_dir) if score >= EVIDENCE_SCORE_THRESHOLD else None
        )
        return SpaiResult(score, evidence_image)
    except subprocess.TimeoutExpired as e:
        raise SpaiInferenceError(
            f"SPAI inference timed out after {settings.spai_timeout_seconds}s"
        ) from e
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
