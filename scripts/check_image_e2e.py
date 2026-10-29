"""실행 중인 서버에 이미지를 보내 가짜정보 탐지가 실제로 나오는지 확인한다(서버를 통한 끝에서 끝까지).

회귀 테스트(tests/regression)는 판정 함수를 직접 부르고 모델을 한 번만 올려 둔다. 실서버는 요청마다 새 프로세스로
돌고 설정도 환경변수로 정해져서, 회귀 테스트가 통과해도 서버는 다른 결과를 낼 수 있다(2026-10-13: 테스트는 탐지
21/30인데 서버는 옛 NLI 모델이라 이 이미지를 못 잡았다). 이 스크립트는 사용자 앱과 같은 길(POST /process/image)로 확인한다.

서버(detection-api 환경)나 text-extraction 환경 어디서든 돈다 - 표준 라이브러리만 쓴다.
사용: python check_image_e2e.py --url http://localhost:8000 --image ..\\tests\\samples\\misinfo_test_image.png --expect 선풍기
성공하면 종료 코드 0, 실패하면 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_IMAGE = Path(__file__).resolve().parent.parent / "tests" / "samples" / "misinfo_test_image.png"
DEFAULT_EXPECT = "선풍기"
# 서버 쪽 분석 제한(가짜정보 300초 등)보다 넉넉히 - 이 스크립트가 먼저 포기하지 않게.
DEFAULT_TIMEOUT_SECONDS = 600


def build_multipart(field: str, filename: str, content_type: str, data: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    body = head + data + f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def evaluate(response: dict, expect: str) -> tuple[bool, list[str]]:
    """응답에서 expect가 들어간 문장이 반박(탐지)으로 나왔는지 본다. (성공 여부, 사람이 읽을 줄들)."""
    lines: list[str] = []
    detection = response.get("misinformation_detection")
    if detection is None:
        return False, ["실패: 응답에 misinformation_detection이 없습니다(문장이 추출되지 않았거나 판정이 실행되지 않음)."]

    claims = detection.get("claims", [])
    lines.append(f"위키 스냅샷: {detection.get('wiki_snapshot')}  /  모델: {detection.get('model')}  /  탐지된 문장 {len(claims)}개")
    for claim in claims:
        lines.append(f"  - 문장: {claim['sentence']}")
        lines.append(f"    이유: {claim['reason']}")
        for evidence in claim["evidence"]:
            lines.append(f"    근거: ({evidence['title']}) {evidence['text']}  {evidence['url']}")

    if any(expect in claim["sentence"] for claim in claims):
        return True, lines
    lines.append(f"실패: '{expect}'가 들어간 문장이 거짓으로 탐지되지 않았습니다(탐지되지 않았거나 다른 문장만 탐지됨).")
    return False, lines


def request_headers(content_type: str) -> dict[str, str]:
    """서버가 DETECTION_API_KEY로 보호돼 있으면(2026-10-28~) 같은 키를 X-Detection-Key 헤더로 보낸다 - 서버를 켠 창과
    같은 환경변수를 읽는다. 비어 있으면 헤더 없이 보낸다(키 검사를 끈 로컬 개발 서버)."""
    headers = {"Content-Type": content_type}
    api_key = os.environ.get("DETECTION_API_KEY", "")
    if api_key:
        headers["X-Detection-Key"] = api_key
    return headers


def run(url: str, image: Path, expect: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> tuple[bool, list[str]]:
    content_type = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    body, header = build_multipart("file", image.name, content_type, image.read_bytes())
    request = urllib.request.Request(
        f"{url.rstrip('/')}/process/image", data=body, headers=request_headers(header), method="POST"
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            payload = json.loads(response.read())
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - start
        return False, [f"걸린 시간: {elapsed:.1f}초", f"실패: 서버가 HTTP {e.code}를 돌려줬습니다: {e.read().decode('utf-8', 'replace')[:500]}"]
    except (urllib.error.URLError, OSError) as e:
        return False, [f"실패: 서버에 연결할 수 없습니다({url}): {e}"]

    elapsed = time.perf_counter() - start
    ok, lines = evaluate(payload, expect)
    ai = payload.get("ai_detection") or {}
    return ok, [f"걸린 시간: {elapsed:.1f}초 (HTTP {status})", f"AI 판독 점수: {ai.get('score')}"] + lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--expect", default=DEFAULT_EXPECT, help="거짓으로 탐지돼야 하는 문장에 들어 있는 단어")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    ok, lines = run(args.url, args.image, args.expect, args.timeout)
    print("\n".join(lines))
    print("\n결과: " + ("성공" if ok else "실패"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
