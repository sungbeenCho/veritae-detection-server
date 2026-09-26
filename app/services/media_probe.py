import subprocess
import tempfile
from pathlib import Path, PureWindowsPath

# ffprobe 실행 자체는 길이만 재는 가벼운 작업이라 모델 추론용 타임아웃(수백 초)과 달리
# 짧게 잡는다 - 이 안에 안 끝나면 길이를 모르는 것으로 취급하고 기존 파이프라인을 그대로
# 진행시킨다(길이를 모르면 판단하지 않는다는 원칙, scam_runner.py의 best-effort 원칙과 동일).
FFPROBE_TIMEOUT_SECONDS = 30


def _safe_filename(filename: str) -> str:
    # PureWindowsPath treats both / and \ as separators, so this strips any
    # directory components regardless of host OS (scam_runner.py/antideepfake_runner.py와 동일 로직).
    name = PureWindowsPath(filename).name
    return name if name and name not in (".", "..") else "upload"


def probe_audio_duration_seconds(audio_bytes: bytes, filename: str) -> float | None:
    """ffprobe로 오디오 길이(초)를 잰다. ffprobe 실행 자체가 실패하거나(returncode != 0),
    출력이 숫자가 아니거나("N/A" 등), 타임아웃이 나면 길이를 알 수 없는 것으로 보고 None을
    반환한다 - 이 경우 호출자는 길이 제한 검사를 건너뛰고 기존 모델 파이프라인을 그대로
    진행시켜야 한다(파일이 실제로 깨졌다면 그 파이프라인이 원래대로 502로 실패한다)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_file = Path(tmp_dir) / _safe_filename(filename)
        input_file.write_bytes(audio_bytes)

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(input_file),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=FFPROBE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return None

        if result.returncode != 0:
            return None

        try:
            return float(result.stdout.strip())
        except ValueError:
            return None
