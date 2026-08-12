import csv
import shutil
import subprocess
import uuid

from app.config import get_settings

SPAI_SCORE_TAG = "spai"


class SpaiInferenceError(RuntimeError):
    pass


def run_spai_inference(image_bytes: bytes, filename: str) -> float:
    settings = get_settings()
    job_dir = settings.work_dir / uuid.uuid4().hex
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    (input_dir / filename).write_bytes(image_bytes)

    command = [
        settings.spai_python,
        "-m", "spai", "infer",
        "--cfg", settings.spai_cfg,
        "--model", settings.spai_model,
        "--input", str(input_dir),
        "--output", str(output_dir),
        "--tag", SPAI_SCORE_TAG,
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

        return float(row[SPAI_SCORE_TAG])
    except subprocess.TimeoutExpired as e:
        raise SpaiInferenceError(
            f"SPAI inference timed out after {settings.spai_timeout_seconds}s"
        ) from e
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
