import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)

# 하위 프로세스가 stderr로 내보내는 한글 로그를 utf-8로 읽는다 - scam_runner와 같은 이유(Windows 콘솔 기본은 cp949).
# 모델 로드 진행 막대("Loading weights: 100%|███")는 서버 로그에 필요 없어 끈다.
_SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_NO_PROGRESS_BAR": "1",
}
# 문장 수가 많아도 서버 로그가 한 번에 너무 길어지지 않게 끝부분만 남긴다.
_SUBPROCESS_LOG_TAIL_CHARS = 6000


class MisinfoInferenceError(RuntimeError):
    pass


class MisinfoResult:
    def __init__(self, model: str, wiki_snapshot: str, claims: list[dict]):
        self.model = model
        self.wiki_snapshot = wiki_snapshot
        self.claims = claims


def _check_nli_model(nli_model: Path) -> None:
    """NLI 모델은 반박이 나온 문장에서 처음 읽는다(지연 로딩). 폴더가 없으면 요청 도중에, 반박이 있을 때만
    늦게 실패해 원인 찾기가 어렵다 - 요청 시작 시점에 경로가 담긴 명확한 오류로 바로 실패시킨다.
    옛 모델로 몰래 넘어가는 대체(폴백)는 하지 않는다 - 그러면 탐지가 조용히 약해진다."""
    if not (nli_model.is_dir() and (nli_model / "config.json").is_file()):
        raise MisinfoInferenceError(
            f"NLI 모델 폴더를 찾을 수 없습니다: {nli_model} "
            "(scripts/train_nli.py로 만들거나 환경변수 MISINFO_NLI_MODEL로 경로를 지정하세요 - docs/SETUP.md 참고)"
        )


def _check_speaker_model(settings) -> None:
    """분류기를 쓰는 판정 방식이면 모델 폴더를 요청 시작 시점에 확인한다(NLI 모델과 같은 이유)."""
    if settings.misinfo_judge_mode in ("speaker", "split"):
        folder = settings.misinfo_speaker_model
        if not (folder.is_dir() and (folder / "config.json").is_file()):
            raise MisinfoInferenceError(
                f"말한 사람 판별 모델 폴더를 찾을 수 없습니다: {folder} "
                "(scripts/train_nli.py --extra-data로 만들거나 MISINFO_SPEAKER_MODEL로 경로 지정, "
                "이전 판정으로 되돌리려면 MISINFO_JUDGE_MODE=current)"
            )


def run_misinfo_inference(sentences: list[str]) -> MisinfoResult:
    """GPU 대기열 자리는 잡지 않는다 - 호출자(text_judge)가 사기 판정과 묶어 자리 하나 안에서 부른다(2026-10-28).
    따로 잡으면 사기 판정이 올린 Ollama 모델이 GPU에 남은 채 자리가 비어, 다른 요청의 Whisper가 같이 올라갈 수 있다."""
    settings = get_settings()
    _check_nli_model(settings.misinfo_nli_model)
    _check_speaker_model(settings)
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
    if settings.misinfo_wiki_search:
        command.append("--wiki-search")
    command += ["--judge-mode", settings.misinfo_judge_mode]
    if settings.misinfo_judge_mode in ("speaker", "split"):
        command += ["--speaker-model", str(settings.misinfo_speaker_model)]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=settings.misinfo_timeout_seconds,
            env=_SUBPROCESS_ENV,
        )

        # 성공해도 남긴다 - 단계별 시간과 경고(LLM 형식 오류로 문장을 판단불가 처리 등)가 여기에만 있다.
        if result.stderr:
            logger.info("misinfo_infer 로그:\n%s", result.stderr[-_SUBPROCESS_LOG_TAIL_CHARS:])

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
