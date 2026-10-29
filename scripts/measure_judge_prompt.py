"""판정 지시문 두 가지를 같은 판정 입력에 돌려 비교하는 측정 도구. 데스크탑에서 한 번 돌린다(Ollama·NLI 필요).

배경(2026-10-17): 화면에는 판정 이유 + 근거 문장 1개 + 문서 링크만 나가는데, 판정 지시문은 근거 문단 번호를 "모두" 고르고
이유를 고른 문단들로 쓰게 되어 있어, 4B가 여러 문서 내용을 섞어 이유를 쓴다(10-01 세종-고려 → 태종, 고래 → 고래상어,
10-16 고래 → 우익 권위주의). 판정 층은 9월부터 바뀐 적이 없다. 후보 A는 "가장 직접 알려주는 문단 1개를 고르고, 이유는 그
문단만으로" 쓰게 한다. 위험: 판정 뒤집힘(지시문 한 줄에 20건 중 4건이 바뀐 전례), 두 문단을 합쳐야 하는 판정(부산 임시 수도),
가설 문단과 평가 문단이 같이 있을 때 평가를 따르는 규칙(선풍기 사망설).

서버 판정 함수(misinfo_infer.judge_sentence)를 그대로 쓰고 판정 지시문만 바꾼다. 검색·재정렬은 문장마다 한 번만 해서 세 번의
판정이 같은 입력을 받는다. 지금 지시문을 두 번 돌려(지금, 지금2) temperature 0에서도 생기는 판정 흔들림을 기준선으로 잰다 -
A에서 바뀐 판정이 이 기준선보다 많아야 지시문 탓이다. 서버 코드는 바꾸지 않는다.

사용(text-extraction 환경, C:\\ai\\veritae-detection-server\\scripts):
  python measure_judge_prompt.py --wiki-index ..\\data\\wiki_index.sqlite3 --ollama-url http://localhost:11434 --ollama-model qwen3.5:4b --gold ..\\tests\\regression\\retrieval_gold.json --dump ..\\judge_prompt_dump.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
import traceback
from pathlib import Path

import misinfo_infer
import misinfo_lib
from kiwipiepy import Kiwi
from measure_retrieval import CASE_FILES, CASES_DIR, load_cases
from misinfo_timing import StepTimer
from nli_check import NliModel, default_model_path

_OLD_IDS = "2. evidence_ids: core_claim이 참인지 거짓인지를 알려주는 근거 문단의 번호를 모두 적는다. 그런 문단이 없으면 빈 목록으로 둔다."
_NEW_IDS = (
    "2. evidence_ids: core_claim이 참인지 거짓인지를 가장 직접 알려주는 근거 문단 하나의 번호만 적는다. "
    "그런 문단이 없으면 빈 목록으로 둔다."
)
_OLD_REASON = "3. reason: 고른 근거 문단이 core_claim에 대해 무엇이라고 말하는지 한 문장으로 설명한다."
_NEW_REASON = (
    "3. reason: 고른 근거 문단 하나가 core_claim에 대해 무엇이라고 말하는지 한 문장으로 설명한다. "
    "다른 문단의 내용은 섞지 않는다."
)
assert _OLD_IDS in misinfo_lib.JUDGE_TEMPLATE and _OLD_REASON in misinfo_lib.JUDGE_TEMPLATE
TEMPLATE_A = misinfo_lib.JUDGE_TEMPLATE.replace(_OLD_IDS, _NEW_IDS).replace(_OLD_REASON, _NEW_REASON)
PROMPTS = {"지금": misinfo_lib.JUDGE_TEMPLATE, "A": TEMPLATE_A, "지금2": misinfo_lib.JUDGE_TEMPLATE}

# 검색·재정렬은 문장마다 한 번만 하고 세 판정이 같이 쓴다 - 시간을 아끼고, 판정 입력이 같다는 것을 보장한다.
_search_cache: dict = {}
_rerank_cache: dict = {}
_original_search, _original_rerank = misinfo_infer.search, misinfo_infer.rerank


def _cached_search(conn, keywords, limit=50):
    key = (keywords, limit)
    if key not in _search_cache:
        _search_cache[key] = _original_search(conn, keywords, limit=limit)
    return _search_cache[key]


def _cached_rerank(sentence, candidates, top_k):
    key = (sentence, top_k)
    if key not in _rerank_cache:
        _rerank_cache[key] = _original_rerank(sentence, candidates, top_k)
    return _rerank_cache[key]


misinfo_infer.search, misinfo_infer.rerank = _cached_search, _cached_rerank


def run_one(kiwi, conn, sentence: str, args, nli, template: str) -> dict:
    """서버 판정 함수를 template 지시문으로 한 번 돌리고, 화면에 안 나가는 판정 결과까지 기록한다."""
    seen: dict = {}
    original_ask, original_fit = misinfo_infer.ask_ollama, misinfo_infer.fit_blocks

    def build_prompt(claim, blocks):
        return template.format(premise=misinfo_lib._format_blocks(blocks), hypothesis=claim)

    def ask(url, model, prompt, schema):
        raw = original_ask(url, model, prompt, schema)
        seen["raw"] = raw
        return raw

    def fit(blocks, *a, **kw):
        kept = original_fit(blocks, *a, **kw)
        seen["blocks"] = kept
        return kept

    misinfo_infer.build_judge_prompt, misinfo_infer.ask_ollama, misinfo_infer.fit_blocks = build_prompt, ask, fit
    timer = StepTimer()
    try:
        shown = misinfo_infer.judge_sentence(kiwi, conn, sentence, args.ollama_url, args.ollama_model, args.evidence_count, nli,
                                             timer=timer)
    except Exception:
        # 문장 하나의 시간 초과 등으로 전체 측정이 멈추지 않게 기록하고 넘어간다.
        shown, timer.outcome = None, "error"
        seen["error"] = traceback.format_exc(limit=2)
    finally:
        misinfo_infer.build_judge_prompt, misinfo_infer.ask_ollama, misinfo_infer.fit_blocks = (
            misinfo_lib.build_judge_prompt, original_ask, original_fit,
        )
    blocks = seen.get("blocks", [])
    verdict = misinfo_lib.parse_judge_response(seen["raw"], len(blocks)) if "raw" in seen else None
    # 문단 본문까지 남긴다 - 이유가 인용 문단 밖(같은 문서의 다른 문단 포함) 내용을 섞었는지 사람이 대조해야 한다.
    record = {"outcome": timer.outcome, "seconds": timer.seconds, "blocks": [[t, b] for t, b in blocks], "shown": shown}
    if "error" in seen:
        record["error"] = seen["error"]
    if verdict:
        titles = [t for t, _ in blocks]
        cited = [titles[i - 1] for i in verdict["evidence_ids"]]
        reason = misinfo_lib.replace_block_numbers(verdict["reason"], blocks)
        # 이유에 인용하지 않은 문서의 제목이 나오면 다른 문서 내용을 섞었다는 신호(보조 지표 - 제목 없이 내용만 섞거나 같은
        # 문서의 다른 문단을 섞은 경우는 못 잡으므로 사람 채점이 기준이다).
        other = sorted({t for t in titles if t not in cited and t in reason})
        record.update(label=verdict["label"], core_claim=verdict["core_claim"], evidence_ids=verdict["evidence_ids"], cited=cited,
                      reason=reason, uncited_in_reason=other)
    return record


def describe(name: str, r: dict) -> str:
    if "label" not in r:
        return f"  {name}: {r['outcome']}"
    ids = ",".join(str(i) for i in r["evidence_ids"])
    text = f"  {name}: {r['label']} | 인용[{ids}]: {', '.join(r['cited'])} | 핵심: {r['core_claim']} | 이유: {r['reason']}"
    if r["shown"]:
        ev = r["shown"]["evidence"][0]
        text += f" | 화면 근거: ({ev['title']}) {ev['text']}"
    if r["uncited_in_reason"]:
        text += f" | 이유에 인용 밖 문서: {', '.join(r['uncited_in_reason'])}"
    return text


def change_kind(expected: str, before: dict, after: dict) -> str:
    """화면 기준으로 판정 변화의 무게를 나눈다."""
    if "error" in (before["outcome"], after["outcome"]):
        return "오류"
    was, now = bool(before["shown"]), bool(after["shown"])
    if was == now:
        return "화면 같음" if before.get("label") == after.get("label") else "판정만 바뀜(화면 무영향)"
    if expected == "contradiction":
        return "탐지 잃음" if was else "탐지 생김"
    return "위험 오답 사라짐" if was else "위험 오답 생김"


def summarize(rows: list[dict], topics: dict[str, str]) -> None:
    print()
    print("===== [2/2] 요약 =====")
    for name in PROMPTS:
        res = [(row["expected"], row[name]) for row in rows]
        false_all = sum(1 for e, _ in res if e == "contradiction")
        detected = sum(1 for e, r in res if e == "contradiction" and r["shown"])
        risky = [row["sentence"] for row in rows if row["expected"] != "contradiction" and row[name]["shown"]]
        outcomes: dict[str, int] = {}
        for _, r in res:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        judged = [r for _, r in res if "label" in r]
        multi = sum(1 for r in judged if len(r["cited"]) >= 2)
        mixed = sum(1 for r in judged if r["uncited_in_reason"])
        judge_seconds = sum(r["seconds"].get("judge_llm", 0.0) for _, r in res)
        print(f"{name}: 탐지(거짓 문장을 띄움) {detected}/{false_all} | 위험 오답(거짓 아닌 문장을 띄움) {len(risky)}"
              f" | 인용 2개 이상 {multi}/{len(judged)} | 이유에 인용 밖 문서 제목 {mixed} | 판정 LLM 합계 {judge_seconds:.0f}초")
        print("  결과 분포: " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())))
        if risky:
            print("  위험 오답 문장: " + " / ".join(risky))
    for left, right in (("지금", "지금2"), ("지금", "A")):
        label = "기준선(같은 지시문 두 번)" if right == "지금2" else "후보 A"
        changed = [row for row in rows if row[left].get("label", row[left]["outcome"]) != row[right].get("label", row[right]["outcome"])]
        kinds: dict[str, int] = {}
        for row in changed:
            kind = change_kind(row["expected"], row[left], row[right])
            kinds[kind] = kinds.get(kind, 0) + 1
        print(f"{label} - {left} -> {right} 판정이 바뀐 문장 {len(changed)}개: " + ", ".join(f"{k} {v}" for k, v in kinds.items()))
        for row in changed:
            print(f"  [{row['expected']}] {row['sentence']} : {row[left].get('label', row[left]['outcome'])} -> "
                  f"{row[right].get('label', row[right]['outcome'])} ({change_kind(row['expected'], row[left], row[right])})")
    if topics:
        # 같은 주제의 변형 문장이 숫자를 부풀리지 않게, 주제마다 탐지한 문장이 하나라도 있는지로도 센다.
        for name in PROMPTS:
            by_topic: dict[str, bool] = {}
            for row in rows:
                topic = topics.get(row["sentence"])
                if topic and row["expected"] == "contradiction":
                    by_topic[topic] = by_topic.get(topic, False) or bool(row[name]["shown"])
            print(f"주제 단위 탐지({name}): {sum(by_topic.values())}/{len(by_topic)}")
    same_input = sum(1 for row in rows if row["지금"]["blocks"] == row["A"]["blocks"] == row["지금2"]["blocks"])
    errors = sum(1 for row in rows for name in PROMPTS if row[name]["outcome"] == "error")
    print(f"세 판정의 입력이 같았던 문장 {same_input}/{len(rows)} (같아야 정상), 오류 {errors}건")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-index", required=True, type=Path)
    parser.add_argument("--ollama-url", required=True)
    parser.add_argument("--ollama-model", required=True)
    parser.add_argument("--dump", required=True, type=Path, help="문장별 전체 기록을 저장할 JSON 경로(문장마다 갱신)")
    parser.add_argument("--cases", nargs="*", type=Path, help="사례 파일(기본: tests/regression의 사례 파일 전부)")
    parser.add_argument("--gold", type=Path, help="정답표(retrieval_gold.json) - 주면 주제 단위로도 집계한다")
    parser.add_argument("--evidence-count", type=int, default=5)
    parser.add_argument("--nli-model", default=default_model_path())
    parser.add_argument("--limit", type=int, help="앞에서 몇 문장만(시험용)")
    args = parser.parse_args()

    paths = args.cases or [CASES_DIR / name for name in CASE_FILES if (CASES_DIR / name).exists()]
    cases, seen_sentences = [], set()
    # 같은 문장은 한 번만(판정은 문장마다 같다). 파일마다 기대값이 다르면 먼저 나온 것을 쓴다 - HEAD 기준 그런 문장은
    # "백신은 자폐증을 유발한다."(contradiction 3번, neutral 1번) 하나다.
    for case in load_cases(paths):
        if case["sentence"] not in seen_sentences:
            seen_sentences.add(case["sentence"])
            cases.append(case)
    cases = cases[: args.limit] if args.limit else cases
    topics = {g["sentence"]: g["topic"] for g in json.loads(args.gold.read_text(encoding="utf-8"))} if args.gold else {}
    conn = sqlite3.connect(f"file:{args.wiki_index.resolve().as_posix()}?mode=ro", uri=True)
    kiwi = Kiwi()
    nli = NliModel(args.nli_model).contradiction_scores

    # 화면 출력만 복사해 보내도 분석할 수 있게, 세 판정 결과를 문장마다 바로 아래 줄에 붙여 쓴다.
    print("===== [1/2] 문장별 결과 =====")
    rows = []
    start = time.perf_counter()
    names = list(PROMPTS)
    try:
        for index, case in enumerate(cases, start=1):
            row = {"sentence": case["sentence"], "expected": case.get("expected", ""), "file": case["file"]}
            # 실행 순서를 문장마다 돌려 바로 앞 호출의 영향이 한 지시문에만 쏠리지 않게 한다.
            order = names[index % 3:] + names[: index % 3]
            for name in order:
                row[name] = run_one(kiwi, conn, case["sentence"], args, nli, PROMPTS[name])
            rows.append(row)
            print(f"{index}/{len(cases)} [{row['expected']}] {row['sentence']}", flush=True)
            for name in names:
                print(describe(name, row[name]), flush=True)
            args.dump.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        misinfo_infer.unload_ollama_model(args.ollama_url, args.ollama_model)
    elapsed = time.perf_counter() - start
    print(f"\n문장 {len(rows)}개, {elapsed:.0f}초 (문장당 {elapsed / max(len(rows), 1):.1f}초)")
    summarize(rows, topics)
    print(f"전체 기록: {args.dump}")


if __name__ == "__main__":
    main()
