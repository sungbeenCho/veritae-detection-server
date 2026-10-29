"""실행 중인 서버에 사기·정상 파일 4개를 보내 사기 판정(LLM v3)이 실제로 나오는지 확인한다(서버를 통한 끝에서 끝까지).

사기 판정을 Lilju에서 로컬 LLM으로 바꾼 뒤(2026-10-28) 확인용. 앱과 같은 길(POST /process/image|audio|video)로 보내고,
다 끝나면 `ollama ps`로 모델이 내려갔는지 본다(사기·가짜정보 판정 뒤에 모델을 내려야 다음 요청의 Whisper와 같이 GPU에
올라가지 않는다). GPU 메모리 최고치는 서버 콘솔의 `gpu_queue_released ... gpu_mem_peak_mb=..` 줄에서 본다.

서버(detection-api 환경)나 text-extraction 환경 어디서든 돈다 - 표준 라이브러리만 쓴다.
사용: python check_scam_e2e.py [--url http://localhost:8000]
모두 기대대로면 종료 코드 0, 아니면 1.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from check_image_e2e import DEFAULT_TIMEOUT_SECONDS, build_multipart, request_headers

SAMPLES = Path(__file__).resolve().parent.parent / "tests" / "samples"
# (이름, 경로, 엔드포인트, content-type, 사기여야 하는지)
CASES = [
    ("사기 문자 캡처", SAMPLES / "scam_test_image.png", "image", "image/png", True),
    ("정상 문자 캡처", SAMPLES / "misinfo_test_image.png", "image", "image/png", False),
    ("사기 음성", SAMPLES / "scam_test_audio.mp3", "audio", "audio/mpeg", True),
    ("사기 영상", SAMPLES / "scam_test_video.mp4", "video", "video/mp4", True),
]
MODEL_SUFFIX = "-scam-v3"


def post(url: str, endpoint: str, path: Path, content_type: str, timeout: float) -> tuple[dict | None, str]:
    body, header = build_multipart("file", path.name, content_type, path.read_bytes())
    request = urllib.request.Request(
        f"{url.rstrip('/')}/process/{endpoint}", data=body, headers=request_headers(header), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read()), ""
    except urllib.error.HTTPError as e:
        return None, f"서버가 HTTP {e.code}를 돌려줬습니다: {e.read().decode('utf-8', 'replace')[:500]}"
    except (urllib.error.URLError, OSError) as e:
        return None, f"서버에 연결할 수 없습니다({url}): {e}"


def check(payload: dict, should_be_scam: bool) -> tuple[bool, list[str]]:
    scam = payload.get("scam_detection")
    if scam is None:
        return False, ["  실패: scam_detection이 없습니다(글자가 추출되지 않았음)."]
    score = scam.get("score")
    reason = scam["evidence"][0]["sentence"] if scam.get("evidence") else "(없음)"
    lines = [f"  모델: {scam.get('model')}  /  사기 점수: {score}", f"  이유: {reason}"]
    problems = []
    if not str(scam.get("model", "")).endswith(MODEL_SUFFIX):
        problems.append(f"모델 이름이 {MODEL_SUFFIX}로 끝나지 않음(서버가 새 코드가 아님)")
    if should_be_scam and score < 0.5:
        problems.append("사기인데 점수가 0.5 미만")
    if not should_be_scam and score >= 0.5:
        problems.append("정상인데 점수가 0.5 이상")
    if not scam.get("evidence"):
        problems.append("이유(근거 칸)가 비어 있음")
    lines += [f"  실패: {p}" for p in problems]
    return not problems, lines


def ollama_loaded() -> str:
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, encoding="utf-8", timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return f"(ollama ps 실행 실패: {e})"
    rows = [line for line in out.stdout.splitlines()[1:] if line.strip()]
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    all_ok = True
    for name, path, endpoint, content_type, should_be_scam in CASES:
        start = time.perf_counter()
        payload, error = post(args.url, endpoint, path, content_type, args.timeout)
        print(f"[{name}] {path.name} - {time.perf_counter() - start:.1f}초")
        if payload is None:
            print(f"  실패: {error}")
            all_ok = False
            continue
        ok, lines = check(payload, should_be_scam)
        print("\n".join(lines))
        all_ok = all_ok and ok

    loaded = ollama_loaded()
    if loaded.startswith("(ollama ps"):
        print(f"\n[모델 내리기] {loaded}")
        all_ok = False
    elif loaded:
        print(f"\n[모델 내리기] 실패: 분석이 끝났는데 모델이 아직 올라가 있습니다\n{loaded}")
        all_ok = False
    else:
        print("\n[모델 내리기] 성공: 올라가 있는 모델 없음")

    print("\n결과: " + ("성공" if all_ok else "실패"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
