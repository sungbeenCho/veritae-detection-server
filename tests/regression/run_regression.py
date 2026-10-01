"""회귀 테스트셋을 실제 위키 인덱스+Ollama로 돌려 정확도/시간/GPU 큐 대기시간을 잰다.
pytest가 아니라 독립 스크립트다 - 실제 모델과 인덱스가 있는 데스크탑에서만 돈다.
프롬프트(scripts/misinfo_lib.py의 PROMPT_TEMPLATE)나 설정값을 바꿀 때마다 이걸로 재확인한다.

기본 실행 대상은 cases_e2e.json(20건, 실제 위키 검색 기반 파이프라인 그대로 돌리는 세트),
cases_grounding.json(14건, 전해 들은 말/OCR 오타/속설을 소개만 하는 참인 문장/근거 두 개를
이어 봐야 하는 문장), cases_heldout.json(25건, 프롬프트를 고칠 때 보지 않는 확인용 세트 -
앞의 두 세트에만 맞춰진 수정인지 가려낸다)이다.

반박 판정은 라벨뿐 아니라 화면에 나가는 근거 문장도 채점한다. 반박이어야 하는 문장에는
evidence_keys(단어 묶음 목록)가 있고, 표시된 근거 문장이 모든 묶음에서 단어를 하나 이상 포함해야
맞는 근거로 본다(2026-10-01: 라벨은 맞는데 "지구의 베어링" 같은 무관한 문장이 근거로 나간 사례).
BAD-EVIDENCE로 찍힌 문장은 자동 채점이 놓친 표현일 수도 있으니 사람이 직접 읽어 확인한다.

cases_basic.json/cases_hard.json은 원래 위키 검색이
아니라 premise(근거 문단)가 함께 주어지는 NLI 분류기 실험 데이터였는데, 이 브랜치로 옮겨오며
premise가 빠져 실제 검색 기반 파이프라인과는 맞지 않는다(자세한 사유는 이 디렉터리의
README.md 참고) - 그래서 기본 실행에서 빼고 --include-nli-legacy를 줬을 때만 참고용으로 돈다.

사용:
  python run_regression.py --wiki-index <db경로> --ollama-url http://localhost:11434 --ollama-model qwen3.5:4b
  python run_regression.py ... --include-nli-legacy   # cases_basic/cases_hard도 참고용으로 같이 돌림
  python run_regression.py ... --llm-cache cache.json   # 같은 프롬프트의 LLM 응답을 재사용(temperature 0)
"""
import argparse
import hashlib
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from kiwipiepy import Kiwi  # noqa: E402
import misinfo_infer  # noqa: E402
from misinfo_infer import judge_sentence  # noqa: E402
from nli_check import NliModel, default_model_path  # noqa: E402

HERE = Path(__file__).parent


def load_cases(name: str) -> list[dict]:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def evidence_ok(text: str, keys: list[list[str]]) -> bool:
    return all(any(word in text for word in group) for group in keys)


def use_llm_cache(path: Path) -> None:
    """temperature 0이라 같은 프롬프트면 같은 응답이 나온다 - 프롬프트를 고친 단계만 다시 묻는다."""
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    ask = misinfo_infer.ask_ollama

    def cached_ask(url, model, prompt, schema):
        key = hashlib.sha256(json.dumps([model, prompt, schema], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if key not in cache:
            cache[key] = ask(url, model, prompt, schema)
            path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        return cache[key]

    misinfo_infer.ask_ollama = cached_ask


def run_group(name: str, cases: list[dict], kiwi, conn, ollama_url: str, model: str, evidence_count: int, nli) -> None:
    correct = 0
    dangerous = 0
    bad_evidence = 0
    slowest = 0.0
    start = time.time()
    for case in cases:
        t0 = time.time()
        claim = judge_sentence(kiwi, conn, case["sentence"], ollama_url, model, evidence_count, nli)
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
                good = "evidence_keys" in case and evidence_ok(evidence["text"], case["evidence_keys"])
                bad_evidence += int(not good)
                print(f"    {'근거' if good else '[BAD-EVIDENCE] 근거'}: ({evidence['title']}) {evidence['text']}")
    elapsed = time.time() - start
    print(f"\n##### {name}: {correct}/{len(cases)} 정확, 위험 오답 {dangerous}건, 잘못된 근거 문장 {bad_evidence}건, "
          f"총 {elapsed:.0f}s(문장당 {elapsed / len(cases):.1f}s, 가장 느린 문장 {slowest:.1f}s)\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-index", required=True, type=Path)
    parser.add_argument("--ollama-url", required=True)
    parser.add_argument("--ollama-model", required=True)
    parser.add_argument("--evidence-count", type=int, default=5)
    parser.add_argument("--nli-model", default=default_model_path())
    parser.add_argument("--llm-cache", type=Path, help="LLM 응답 캐시 파일(프롬프트 수정 반복 시험용)")
    parser.add_argument(
        "--include-nli-legacy",
        action="store_true",
        help="premise 없이는 이 파이프라인과 맞지 않는 cases_basic/cases_hard(NLI 벤치마크 "
             "데이터)도 참고용으로 같이 돈다. 머지 게이트 아님 - README.md 참고.",
    )
    args = parser.parse_args()

    # 모든 문장의 1차 판정과 재판정 결과(misinfo 로거)는 자세히 보여주고, httpx/HuggingFace의
    # 접속 확인 로그는 끈다(2026-10-01: 그대로 두면 같은 요청이 여러 번 찍혀 결과를 읽기 어려웠다).
    logging.basicConfig(level=logging.WARNING, format="    [%(levelname)s] %(message)s")
    logging.getLogger("misinfo").setLevel(logging.DEBUG)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    if args.llm_cache:
        use_llm_cache(args.llm_cache)
    conn = sqlite3.connect(args.wiki_index)
    kiwi = Kiwi()
    nli = NliModel(args.nli_model).contradiction_scores

    run_group("cases_e2e", load_cases("cases_e2e.json"), kiwi, conn, args.ollama_url, args.ollama_model, args.evidence_count, nli)
    run_group("cases_grounding", load_cases("cases_grounding.json"), kiwi, conn, args.ollama_url, args.ollama_model, args.evidence_count, nli)
    run_group("cases_heldout", load_cases("cases_heldout.json"), kiwi, conn, args.ollama_url, args.ollama_model, args.evidence_count, nli)

    if args.include_nli_legacy:
        run_group("cases_basic (참고용, NLI 벤치마크 - 머지 게이트 아님)", load_cases("cases_basic.json"), kiwi, conn, args.ollama_url, args.ollama_model, args.evidence_count, nli)
        run_group("cases_hard (참고용, NLI 벤치마크 - 머지 게이트 아님)", load_cases("cases_hard.json"), kiwi, conn, args.ollama_url, args.ollama_model, args.evidence_count, nli)


if __name__ == "__main__":
    main()
