import json
import shutil
import subprocess
import uuid
from pathlib import Path

from app.config import get_settings
from app.services.gpu_queue import get_gpu_queue


class MisinfoInferenceError(RuntimeError):
    pass


class MisinfoResult:
    def __init__(self, model: str, wiki_snapshot: str, claims: list[dict]):
        self.model = model
        self.wiki_snapshot = wiki_snapshot
        self.claims = claims


def run_misinfo_inference(sentences: list[str]) -> MisinfoResult:
    settings = get_settings()
    job_dir = settings.text_extraction_work_dir / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    sentences_file = job_dir / "sentences.json"
    output_file = job_dir / "result.json"
    sentences_file.write_text(json.dumps(sentences, ensure_ascii=False), encoding="utf-8")

    command = [
        settings.text_extraction_python,
        str(settings.misinfo_script),
        "--sentences", str(sentences_file),
        "--output", str(output_file),
        "--wiki-index", str(settings.wiki_index_path),
        "--ollama-url", settings.ollama_url,
        "--ollama-model", settings.ollama_model,
        "--evidence-count", str(settings.misinfo_evidence_chunk_count),
        "--nli-model", str(settings.misinfo_nli_model),
    ]

    try:
        with get_gpu_queue().acquire("misinfo_infer"):
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=settings.misinfo_timeout_seconds,
            )

        if result.returncode != 0:
            raise MisinfoInferenceError(f"가짜정보 판정 실패: {result.stderr[-2000:]}")

        if not output_file.exists():
            raise MisinfoInferenceError(f"expected output JSON not found: {output_file}")

        return _parse_result(output_file)
    except subprocess.TimeoutExpired as e:
        raise MisinfoInferenceError(
            f"가짜정보 판정이 {settings.misinfo_timeout_seconds}초 안에 끝나지 않았습니다"
        ) from e
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def _parse_result(output_file: Path) -> MisinfoResult:
    try:
        data = json.loads(output_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        raise MisinfoInferenceError(f"가짜정보 판정 output JSON을 읽거나 파싱할 수 없습니다: {output_file}") from e
    return MisinfoResult(
        model=data.get("model", ""), wiki_snapshot=data.get("wiki_snapshot", ""), claims=data.get("claims", [])
    )
