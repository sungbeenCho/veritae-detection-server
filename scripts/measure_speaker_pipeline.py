"""판정 방식 네 가지를 같은 문장에 돌려 서버 판정 함수 기준으로 비교하는 측정 도구. 데스크탑에서 한 번 돌린다(Ollama·NLI·분류기 필요).

  - 지금: 06ba7c5와 같은 판정(misinfo_infer.judge_sentence 기본값) - 비교 기준.
  - A: 판정 지시문 A만(--judge-mode a). 이유 섞임을 줄인다(2026-10-17 158문장 측정: 섞임 13·12 → 2·5).
  - 분류기v1 / 분류기v2: --judge-mode speaker. 4B가 위키 없이 내용만 정리하고, 말한 사람이 그 내용을 맞다고 하는지 아니라고
    하는지는 분류기(data\\speaker_model / data\\speaker_model_v2)가 정하고, 판정은 그 내용의 참/거짓만 본다(지시문 A).
    두 분류기는 같은 4B 정리·판정 결과를 함께 쓴다(4B를 한 번만 돌린다). 분류기가 판정 뒤의 단계(표시 여부·근거 고르기)만
    바꾸기 때문이다. 둘 다 사실 없음이면 판정을 건너뛴다(서버와 같다).

판단(측정 전에 정함): 지금 대비 위험 오답(틀리지 않은 문장을 띄움)이 줄고 맞는 탐지를 잃지 않아야 한다. 나쁜 탐지(이유·근거가
엉뚱한 탐지)를 잃는 것을 괜찮다고 볼지는 사용자 결정이라 사람 채점으로 따로 센다. 시간·메모리도 함께 본다(300초 상한).
지금 지시문을 두 번 돌린 흔들림은 10-17 측정에서 158문장 0건이라 이번에는 다시 재지 않는다.

사용(text-extraction 환경, C:\\ai\\veritae-detection-server\\scripts):
  python measure_speaker_pipeline.py --wiki-index ..\\data\\wiki_index.sqlite3 --ollama-url http://localhost:11434 --ollama-model qwen3.5:4b --dump ..\\speaker_pipeline_dump.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

import misinfo_infer
import misinfo_lib
from kiwipiepy import Kiwi
from measure_retrieval import CASE_FILES, CASES_DIR, load_cases
from misinfo_timing import StepTimer
from nli_check import NliModel, default_model_path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# 기존 회귀 사례 + 아니라고 한 내용이 참인 문장 5개(10-16) + 학습 데이터를 본 적 없는 작성자가 쓴 새 문장(10-17, 검증 2명).
EXTRA_CASE_FILES = ["cases_false_negation.json", "cases_fresh.json"]
NAMES = ("지금", "A", "분류기v1", "분류기v2")
SPEAKERS = {"분류기v1": "speaker_model", "분류기v2": "speaker_model_v2"}

# 지금·A는 같은 원문으로 검색·재정렬하므로 문장마다 한 번만 한다 - 시간을 아끼고 두 판정의 입력이 같다는 것을 보장한다.
# 캐시에서 꺼낸 쪽도 처음 계산에 든 시간을 낸 것으로 친다(_credit) - 안 그러면 뒤에 도는 방식의 검색 시간이 0이 된다.
_search_cache: dict = {}
_rerank_cache: dict = {}
_credit: dict[str, float] = {}
_original_search, _original_rerank = misinfo_infer.search, misinfo_infer.rerank


def _cached(cache: dict, key, step: str, compute):
    if key in cache:
        value, seconds = cache[key]
        _credit[step] = _credit.get(step, 0.0) + seconds
        return value
    start = time.perf_counter()
    value = compute()
    cache[key] = (value, time.perf_counter() - start)
    return value


def _cached_search(conn, keywords, limit=50):
    return _cached(_search_cache, (keywords, limit), "search", lambda: _original_search(conn, keywords, limit=limit))


def _cached_rerank(sentence, candidates, top_k):
    return _cached(_rerank_cache, (sentence, top_k), "rerank", lambda: _original_rerank(sentence, candidates, top_k))


def _charge(timer: StepTimer) -> None:
    for step, seconds in _credit.items():
        timer.seconds[step] = timer.seconds.get(step, 0.0) + seconds
    _credit.clear()


misinfo_infer.search, misinfo_infer.rerank = _cached_search, _cached_rerank


def run_current(kiwi, conn, sentence: str, args, nli, template: str) -> dict:
    """서버 판정 함수(judge_sentence)를 한 번 돌리고, 화면에 안 나가는 판정 결과까지 기록한다."""
    seen: dict = {}
    original_ask, original_fit = misinfo_infer.ask_ollama, misinfo_infer.fit_blocks

    def ask(url, model, prompt, schema):
        raw = original_ask(url, model, prompt, schema)
        seen["raw"] = raw
        return raw

    def fit(blocks, *a, **kw):
        kept = original_fit(blocks, *a, **kw)
        seen["blocks"] = kept
        return kept

    misinfo_infer.ask_ollama, misinfo_infer.fit_blocks = ask, fit
    timer = StepTimer()
    _credit.clear()
    try:
        shown = misinfo_infer.judge_sentence(kiwi, conn, sentence, args.ollama_url, args.ollama_model, args.evidence_count,
                                             nli.contradiction_scores, timer=timer, judge_template=template)
    except Exception:
        shown, timer.outcome = None, "error"
        seen["error"] = traceback.format_exc(limit=2)
    finally:
        misinfo_infer.ask_ollama, misinfo_infer.fit_blocks = original_ask, original_fit
        _charge(timer)
    blocks = seen.get("blocks", [])
    verdict = misinfo_lib.parse_judge_response(seen["raw"], len(blocks)) if "raw" in seen else None
    record = {"outcome": timer.outcome, "seconds": timer.seconds, "blocks": [[t, b] for t, b in blocks], "shown": shown}
    if "error" in seen:
        record["error"] = seen["error"]
    if verdict:
        record.update(label=verdict["label"], claim=verdict["core_claim"], **_cited(verdict, blocks))
    return record


def _cited(verdict: dict, blocks) -> dict:
    titles = [t for t, _ in blocks]
    cited = [titles[i - 1] for i in verdict["evidence_ids"]]
    reason = misinfo_lib.replace_block_numbers(verdict["reason"], blocks)
    # 이유에 인용하지 않은 문서 제목이 나오면 다른 문서 내용을 섞었다는 신호(보조 지표 - 사람 채점이 기준).
    return {"evidence_ids": verdict["evidence_ids"], "cited": cited, "reason": reason,
            "uncited_in_reason": sorted({t for t in titles if t not in cited and t in reason})}


def run_speaker(kiwi, conn, sentence: str, args, nli, speakers: dict) -> dict[str, dict]:
    """4B 정리·판정은 한 번, 표시 여부·근거는 분류기마다. 서버 judge_sentence_speaker와 같은 함수들을 같은 순서로 쓴다."""
    base = StepTimer()
    _credit.clear()
    out = {name: {"outcome": "unknown", "shown": None} for name in speakers}
    try:
        claim = misinfo_infer.extract_claim(sentence, args.ollama_url, args.ollama_model, base)
        if claim is None or not claim:
            for r in out.values():
                r.update(outcome=base.outcome if claim is None else "no_fact", claim=claim)
            return _with_seconds(out, base, {})
        speaker_seconds, said = {}, {}
        for name, model in speakers.items():
            start = time.perf_counter()
            said[name] = misinfo_infer.speaker_of(model, sentence, claim)
            speaker_seconds[name] = time.perf_counter() - start
        for name in speakers:
            out[name].update(claim=claim, speaker=said[name])
        # 사실 없음으로 본 분류기는 서버에서 여기서 끝난다 - 그쪽 시간·결과에는 뒤의 판정을 넣지 않는다.
        stopped = {name for name, s in said.items() if s == misinfo_lib.SPEAKER_NO_FACT}
        for name in stopped:
            out[name].update(outcome="no_fact_speaker", seconds={**base.seconds, "speaker": speaker_seconds[name]})
        if len(stopped) == len(speakers):
            return out
        active = [name for name in speakers if name not in stopped]
        invented = misinfo_infer.unsupported_words(kiwi, sentence, claim)
        if invented:
            for name in active:
                out[name].update(outcome="extract_unfaithful", invented=invented)
            return _with_seconds(out, base, speaker_seconds, active)
        judged = misinfo_infer.judge_claim(kiwi, conn, claim, args.ollama_url, args.ollama_model, args.evidence_count,
                                           False, base, misinfo_lib.CLAIM_JUDGE_TEMPLATE_A)
        _charge(base)
        if judged is None:
            for name in active:
                out[name]["outcome"] = base.outcome
            return _with_seconds(out, base, speaker_seconds, active)
        blocks, verdict = judged
        for name in active:
            r = out[name]
            r.update(blocks=[[t, b] for t, b in blocks], label=verdict["label"], **_cited(verdict, blocks))
            timer = StepTimer()
            r["shown"] = misinfo_infer.show_if_wrong(sentence, claim, said[name], blocks, verdict, nli.contradiction_scores,
                                                     nli.entailment_scores, timer)
            r["outcome"], r["after_judge_seconds"] = timer.outcome, timer.seconds
        return _with_seconds(out, base, speaker_seconds, active)
    except Exception:
        error = traceback.format_exc(limit=2)
        for r in out.values():
            r.update(outcome="error", shown=None, error=error)
        return _with_seconds(out, base, {})


def _with_seconds(out: dict, base: StepTimer, speaker_seconds: dict, names=None) -> dict:
    """분류기 둘은 4B 정리·판정을 같이 쓰지만, 시간은 서버에서 한 방식만 켰을 때처럼 각자 통째로 센다."""
    for name in names if names is not None else out:
        r = out[name]
        r["seconds"] = {**base.seconds, "speaker": speaker_seconds.get(name, 0.0), **r.pop("after_judge_seconds", {})}
    return out


def gpu_memory_used_mb() -> int | None:
    """GPU 전체 사용량(Ollama 4B 포함). nvidia-smi가 없으면 None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.splitlines()[0].strip())
    except Exception:
        return None


def describe(name: str, r: dict) -> str:
    if "label" not in r:
        extra = f" | 정리: {r['claim']}" if r.get("claim") else ""
        extra += f" | 분류기: {r['speaker']}" if r.get("speaker") else ""
        return f"  {name}: {r['outcome']}{extra}"
    text = f"  {name}: {r['label']}"
    if r.get("speaker"):
        text += f" | 분류기: {r['speaker']}"
    text += f" | 띄움: {'예' if r['shown'] else '아니오'} | 인용: {', '.join(r['cited'])} | 내용: {r['claim']} | 이유: {r['reason']}"
    if r["shown"]:
        ev = r["shown"]["evidence"][0]
        text += f" | 화면 근거: ({ev['title']}) {ev['text']}"
    if r["uncited_in_reason"]:
        text += f" | 이유에 인용 밖 문서: {', '.join(r['uncited_in_reason'])}"
    return text


def change_kind(expected: str, before: dict, after: dict) -> str:
    if "error" in (before["outcome"], after["outcome"]):
        return "오류"
    was, now = bool(before["shown"]), bool(after["shown"])
    if was == now:
        return "화면 같음"
    if expected == "contradiction":
        return "탐지 잃음" if was else "탐지 생김"
    return "위험 오답 사라짐" if was else "위험 오답 생김"


def summarize(rows: list[dict]) -> None:
    print()
    print("===== [2/2] 요약 =====")
    groups = {"전체": rows, "기존 회귀 사례": [r for r in rows if r["file"] not in EXTRA_CASE_FILES],
              "새 문장(cases_fresh)": [r for r in rows if r["file"] == "cases_fresh.json"],
              "참인 내용을 아니라고 함(cases_false_negation)": [r for r in rows if r["file"] == "cases_false_negation.json"]}
    for group, subset in groups.items():
        if not subset:
            continue
        print(f"[{group}] 문장 {len(subset)}개")
        for name in NAMES:
            false_all = sum(1 for row in subset if row["expected"] == "contradiction")
            detected = sum(1 for row in subset if row["expected"] == "contradiction" and row[name]["shown"])
            risky = [row["sentence"] for row in subset if row["expected"] != "contradiction" and row[name]["shown"]]
            mixed = sum(1 for row in subset if row[name].get("uncited_in_reason"))
            print(f"  {name}: 탐지 {detected}/{false_all} | 위험 오답 {len(risky)} | 이유에 인용 밖 문서 제목 {mixed}")
            if risky:
                print("    위험 오답 문장: " + " / ".join(risky))
    for name in NAMES:
        outcomes: dict[str, int] = {}
        for row in rows:
            outcomes[row[name]["outcome"]] = outcomes.get(row[name]["outcome"], 0) + 1
        steps: dict[str, float] = {}
        for row in rows:
            for step, sec in row[name]["seconds"].items():
                steps[step] = steps.get(step, 0.0) + sec
        print(f"{name} 결과 분포: " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())))
        print(f"{name} 단계별 합계(초): " + ", ".join(f"{k} {v:.0f}" for k, v in steps.items()))
    # 문장 하나를 서버에서 처리하는 시간(한 방식만 켰을 때). 요청마다 드는 고정 비용(프로세스 시작, 모델 로드·내림)은 빠져 있다 -
    # 서버 경로로 따로 잰다.
    for name in NAMES:
        totals = sorted(sum(row[name]["seconds"].values()) for row in rows)
        if totals:
            def pick(q):
                return totals[min(len(totals) - 1, int(q * len(totals)))]
            print(f"{name} 문장당 시간: 중간 {pick(0.5):.1f}초, 상위 5% {pick(0.95):.1f}초, 최대 {totals[-1]:.1f}초, 합계 {sum(totals):.0f}초")
    print("판단(지금 대비, 화면 기준). 탐지 잃음은 지금의 이유·근거가 엉뚱했던 나쁜 탐지인지 사람이 채점한다:")
    for name in NAMES[1:]:
        changed = [row for row in rows if bool(row["지금"]["shown"]) != bool(row[name]["shown"])]
        kinds: dict[str, int] = {}
        for row in changed:
            kind = change_kind(row["expected"], row["지금"], row[name])
            kinds[kind] = kinds.get(kind, 0) + 1
        risky_now = sum(1 for row in rows if row["expected"] != "contradiction" and row["지금"]["shown"])
        risky_new = sum(1 for row in rows if row["expected"] != "contradiction" and row[name]["shown"])
        lost = kinds.get("탐지 잃음", 0)
        verdict = ("통과(위험 오답 줄고 탐지 잃음 0)" if risky_new < risky_now and lost == 0 else
                   "사람 채점 필요(탐지 잃음이 나쁜 탐지뿐이면 통과)" if risky_new < risky_now else "불통과(위험 오답이 줄지 않음)")
        print(f"  {name}: 위험 오답 {risky_now} -> {risky_new}, 화면이 바뀐 문장 {len(changed)}개("
              + ", ".join(f"{k} {v}" for k, v in kinds.items()) + f") => {verdict}")
        for row in changed:
            print(f"    [{row['expected']}] ({row['file']}) {row['sentence']} : {change_kind(row['expected'], row['지금'], row[name])}")
            for side in ("지금", name):
                r = row[side]
                if r["shown"]:
                    ev = r["shown"]["evidence"][0]
                    print(f"      {side} 이유: {r['shown']['reason']} | 근거: ({ev['title']}) {ev['text']}")
    differ = [row for row in rows if row["분류기v1"].get("speaker") != row["분류기v2"].get("speaker")]
    print(f"두 분류기의 말한 사람 판단이 다른 문장 {len(differ)}개")
    for row in differ:
        print(f"  {row['sentence']} || {row['분류기v1'].get('claim')} : v1 {row['분류기v1'].get('speaker')} / v2 {row['분류기v2'].get('speaker')}")
    errors = sum(1 for row in rows for name in NAMES if row[name]["outcome"] == "error")
    same_input = sum(1 for row in rows if row["지금"]["blocks"] == row["A"]["blocks"])
    print(f"지금·A 판정 입력이 같았던 문장 {same_input}/{len(rows)} (같아야 정상), 오류 {errors}건")


def _write_dump(path: Path, rows: list[dict]) -> None:
    try:
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as error:
        print(f"(기록 파일 쓰기 실패 - 측정은 계속: {error})", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-index", required=True, type=Path)
    parser.add_argument("--ollama-url", required=True)
    parser.add_argument("--ollama-model", required=True)
    parser.add_argument("--dump", required=True, type=Path, help="문장별 전체 기록을 저장할 JSON 경로(문장마다 갱신)")
    parser.add_argument("--cases", nargs="*", type=Path, help="사례 파일(기본: 회귀 사례 전부 + 새 문장 + 참인 내용 부정)")
    parser.add_argument("--evidence-count", type=int, default=5)
    parser.add_argument("--nli-model", default=default_model_path())
    parser.add_argument("--limit", type=int, help="앞에서 몇 문장만(시험용)")
    parser.add_argument("--summary-only", action="store_true", help="측정 없이 --dump 기록으로 요약만 다시 낸다(중간에 멈췄을 때)")
    args = parser.parse_args()
    if args.summary_only:
        summarize(json.loads(args.dump.read_text(encoding="utf-8")))
        return
    missing = [str(CASES_DIR / n) for n in CASE_FILES + EXTRA_CASE_FILES if not (CASES_DIR / n).exists()] if not args.cases else []
    missing += [str(DATA_DIR / folder) for folder in SPEAKERS.values() if not (DATA_DIR / folder).exists()]
    if missing:
        parser.error("없는 파일·폴더: " + ", ".join(missing))

    names = CASE_FILES + EXTRA_CASE_FILES
    paths = args.cases or [CASES_DIR / name for name in names if (CASES_DIR / name).exists()]
    cases, seen_sentences = [], set()
    for case in load_cases(paths):
        if case["sentence"] not in seen_sentences:
            seen_sentences.add(case["sentence"])
            cases.append(case)
    if args.limit:
        # 시험 실행에서도 세 묶음이 다 돌게 앞에서부터 고르게 섞는다.
        by_file: dict[str, list] = {}
        for case in cases:
            by_file.setdefault(case["file"], []).append(case)
        picked = []
        while len(picked) < args.limit and any(by_file.values()):
            for file_cases in by_file.values():
                if file_cases and len(picked) < args.limit:
                    picked.append(file_cases.pop(0))
        cases = picked
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["file"]] = counts.get(case["file"], 0) + 1
    print(f"문장 {len(cases)}개: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    conn = sqlite3.connect(f"file:{args.wiki_index.resolve().as_posix()}?mode=ro", uri=True)
    kiwi = Kiwi()
    gpu_before = gpu_memory_used_mb()
    nli = NliModel(args.nli_model)
    # 첫 문장 시간에 모델 로드가 섞이지 않게 NLI와 e5를 미리 올린다(실서버는 요청마다 로드 - 서버 경로로 따로 잰다).
    nli.label_probabilities(["시험 문장이다."], "시험이다.")
    misinfo_infer.embed(["query: 시험"])
    speakers = {}
    load_seconds = {}
    for name, folder in SPEAKERS.items():
        start = time.perf_counter()
        model = NliModel(str(DATA_DIR / folder), load_step=f"load_{folder}")
        model.label_probabilities(["시험 문장이다."], "시험이다.")  # 지금 올려서 로드 시간을 따로 잰다
        speakers[name], load_seconds[name] = model, time.perf_counter() - start
    gpu_loaded = gpu_memory_used_mb()
    print("분류기 로드 시간(측정 프로세스 안, 참고용): " + ", ".join(f"{k} {v:.1f}초" for k, v in load_seconds.items()))
    print(f"GPU 사용량: 모델 올리기 전 {gpu_before} MB, NLI·e5·분류기 둘 올린 뒤 {gpu_loaded} MB (None = 측정 못 함)")

    print("===== [1/2] 문장별 결과 =====")
    rows, gpu_peak, error_streak = [], 0, 0
    start = time.perf_counter()
    try:
        for index, case in enumerate(cases, start=1):
            row = {"sentence": case["sentence"], "expected": case.get("expected", ""), "file": case["file"],
                   "category": case.get("category", "")}
            # 실행 순서를 문장마다 돌려 바로 앞 호출의 영향이 한 방식에만 쏠리지 않게 한다.
            jobs = [("지금", lambda: {"지금": run_current(kiwi, conn, case["sentence"], args, nli, misinfo_lib.JUDGE_TEMPLATE)}),
                    ("A", lambda: {"A": run_current(kiwi, conn, case["sentence"], args, nli, misinfo_lib.JUDGE_TEMPLATE_A)}),
                    ("분류기", lambda: run_speaker(kiwi, conn, case["sentence"], args, nli, speakers))]
            for _, job in jobs[index % 3:] + jobs[: index % 3]:
                row.update(job())
            used = gpu_memory_used_mb()
            if used:
                gpu_peak = max(gpu_peak, used)
            rows.append(row)
            # Ollama가 꺼졌을 때 끝까지 오류만 쌓지 않게, 세 문장 연속 모든 방식이 오류면 멈춘다.
            error_streak = error_streak + 1 if all(row[name]["outcome"] == "error" for name in NAMES) else 0
            print(f"{index}/{len(cases)} [{row['expected']}] ({row['file']}) {row['sentence']}", flush=True)
            for name in NAMES:
                print(describe(name, row[name]), flush=True)
            _write_dump(args.dump, rows)
            if error_streak >= 3:
                print("세 문장 연속 모든 방식이 오류라 멈춘다(Ollama가 켜져 있는지 확인):\n" + row["지금"].get("error", ""))
                break
    finally:
        misinfo_infer.unload_ollama_model(args.ollama_url, args.ollama_model)
    elapsed = time.perf_counter() - start
    peak = f"{gpu_peak} MB(문장 끝마다 잰 값 중 최고, Ollama 4B 포함)" if gpu_peak else "측정 못 함"
    print(f"\n문장 {len(rows)}개, {elapsed:.0f}초 (문장당 {elapsed / max(len(rows), 1):.1f}초), GPU 사용량 최고 {peak}")
    summarize(rows)
    print(f"전체 기록: {args.dump}")


if __name__ == "__main__":
    main()
