"""회귀 테스트셋을 실제 위키 인덱스+Ollama로 돌려 정확도/시간/GPU 큐 대기시간을 잰다.
pytest가 아니라 독립 스크립트다 - 실제 모델과 인덱스가 있는 데스크탑에서만 돈다.
프롬프트(scripts/misinfo_lib.py의 PROMPT_TEMPLATE)나 설정값을 바꿀 때마다 이걸로 재확인한다.

기본 실행 대상은 cases_e2e.json(20건, 실제 위키 검색 기반 파이프라인 그대로 돌리는 세트)과
cases_grounding.json(14건, 전해 들은 말/OCR 오타/속설을 소개만 하는 참인 문장/근거 두 개를
이어 봐야 하는 문장)이다. cases_basic.json/cases_hard.json은 원래 위키 검색이
아니라 premise(근거 문단)가 함께 주어지는 NLI 분류기 실험 데이터였는데, 이 브랜치로 옮겨오며
premise가 빠져 실제 검색 기반 파이프라인과는 맞지 않는다(자세한 사유는 이 디렉터리의
README.md 참고) - 그래서 기본 실행에서 빼고 --include-nli-legacy를 줬을 때만 참고용으로 돈다.

사용:
  python run_regression.py --wiki-index <db경로> --ollama-url http://localhost:11434 --ollama-model qwen3.5:4b
  python run_regression.py ... --include-nli-legacy   # cases_basic/cases_hard도 참고용으로 같이 돌림
"""
import argparse
import json
import logging
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from kiwipiepy import Kiwi  # noqa: E402
from misinfo_infer import judge_sentence  # noqa: E402

HERE = Path(__file__).parent


def load_cases(name: str) -> list[dict]:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def run_group(name: str, cases: list[dict], kiwi, conn, ollama_url: str, model: str, evidence_count: int) -> None:
    correct = 0
    dangerous = 0
    slowest = 0.0
    start = time.time()
    for case in cases:
        t0 = time.time()
        claim = judge_sentence(kiwi, conn, case["sentence"], ollama_url, model, evidence_count)
        took = time.time() - t0
        slowest = max(slowest, took)
        got = "contradiction" if claim is not None else "not_contradiction"
        expected_is_contradiction = case["expected"] == "contradiction"
        ok = (got == "contradiction") == expected_is_contradiction
        danger = expected_is_contradiction is False and got == "contradiction"
        correct += int(ok)
        dangerous += int(danger)
        tag = "OK" if ok else ("DANGER" if danger else "MISS")
        print(f"[{tag}] {case['category']:12s} expected={case['expected']:13s} got={got:16s} "
              f"({took:.1f}s) | {case['sentence']}", flush=True)
        if claim is not None:
            # 반박으로 나온 건은 라벨만 보지 말고 이유와 인용된 근거가 실제로 반박 내용인지
            # 사람이 직접 읽어서 확인한다(2026-10-01: 라벨은 반박인데 근거는 속설을 뒷받침한 사례).
            print(f"    이유: {claim['reason']}")
            for evidence in claim["evidence"]:
                print(f"    근거: ({evidence['title']}) {evidence['text']}")
    elapsed = time.time() - start
    print(f"\n##### {name}: {correct}/{len(cases)} 정확, 위험 오답 {dangerous}건, "
          f"총 {elapsed:.0f}s(문장당 {elapsed / len(cases):.1f}s, 가장 느린 문장 {slowest:.1f}s)\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-index", required=True, type=Path)
    parser.add_argument("--ollama-url", required=True)
    parser.add_argument("--ollama-model", required=True)
    parser.add_argument("--evidence-count", type=int, default=5)
    parser.add_argument(
        "--include-nli-legacy",
        action="store_true",
        help="premise 없이는 이 파이프라인과 맞지 않는 cases_basic/cases_hard(NLI 벤치마크 "
             "데이터)도 참고용으로 같이 돈다. 머지 게이트 아님 - README.md 참고.",
    )
    args = parser.parse_args()

    # 1차 반박이 정밀 재판정에서 뒤집힌 경우 등 판정 과정 로그(INFO/WARNING)를 같이 보여준다.
    logging.basicConfig(level=logging.INFO, format="    [%(levelname)s] %(message)s")
    # 생각 모드(think)와 JSON 스키마(format)를 같이 쓰는 재판정은 Ollama 0.34.4 이전 버전에서
    # 깨진 JSON이나 빈 응답을 낼 수 있다(ollama PR #18479). 결과를 읽기 전에 버전부터 확인한다.
    with urllib.request.urlopen(f"{args.ollama_url}/api/version", timeout=10) as resp:
        print(f"Ollama 버전: {json.loads(resp.read())['version']} (재판정에는 0.34.4 이상 필요)\n")
    conn = sqlite3.connect(args.wiki_index)
    kiwi = Kiwi()

    run_group("cases_e2e", load_cases("cases_e2e.json"), kiwi, conn, args.ollama_url, args.ollama_model, args.evidence_count)
    run_group("cases_grounding", load_cases("cases_grounding.json"), kiwi, conn, args.ollama_url, args.ollama_model, args.evidence_count)

    if args.include_nli_legacy:
        run_group("cases_basic (참고용, NLI 벤치마크 - 머지 게이트 아님)", load_cases("cases_basic.json"), kiwi, conn, args.ollama_url, args.ollama_model, args.evidence_count)
        run_group("cases_hard (참고용, NLI 벤치마크 - 머지 게이트 아님)", load_cases("cases_hard.json"), kiwi, conn, args.ollama_url, args.ollama_model, args.evidence_count)


if __name__ == "__main__":
    main()
