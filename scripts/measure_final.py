"""마지막 확인(2026-10-18): 지금까지 측정에서 나아진 부분만 묶은 판정 방식 세 가지를 지금(06ba7c5)과 비교한다. 데스크탑에서 한 번 돌린다.

  - 지금: 06ba7c5 판정(judge_sentence, 지금 지시문).
  - 나눔: 4B가 정리한 내용을 분류기가 "아니라고 함"으로 보는 문장(부정 문장)만 분류기 방식(정리된 내용으로 검색·판정, 지시문 A,
    맞다고+반박 / 아니라고+지지면 띄움)으로, 나머지는 지금 방식으로 판정한다. 2026-10-18 225문장 기록에서 분류기 방식이 나았던
    문장은 거의 다 "아니라고 함" 문장이었고 놓친 문장은 전부 "맞다고 함" 문장이었다(손계산 위험 오답 19 -> 11, 탐지 80 -> 81).
  - 나눔+A: 나눔의 나머지 문장도 판정 지시문 A(근거 문단 하나만, 이유는 그 문단만)로. 이유 섞임 13·12 -> 2·5(10-17 채점).
  - 나눔+A+제목검색: 위 방식에서 검색만 바꾼다 - 문장(또는 정리된 내용)의 명사와 같은 제목의 위키 문서를 후보에 넣고, 그런 문서
    최대 2개는 상위 5개 안에 자리를 준다(나머지는 지금처럼 e5 순). 검색 측정(10-16)에서 맞는 문서를 놓친 원인 1위가 "단어가 더 많이
    겹치는 엉뚱한 문서와의 점수 경쟁"이었고, 문서 제목은 점수에 쓰지 않았다.

문장: 기존 225문장(어제 기록 speaker_pipeline_dump.json을 다시 쓴다 - 지금·A·분류기 결과는 같은 지시문이면 매번 같다) + 학습·설계에
쓰지 않은 새 문장(cases_fresh2.json). 나눔 규칙은 어제 225문장 결과를 보고 정했으므로 그 문장들의 숫자는 좋게 나오는 게 당연하다 -
판단은 새 문장 결과를 기준으로 한다.

판단 기준(측정 전에 정함): 새 문장에서 위험 오답(틀리지 않은 문장을 띄움)이 지금보다 적고 탐지(틀린 문장을 띄움)가 지금보다
적지 않으며, 기존 225문장에서도 위험 오답이 지금보다 많지 않고 탐지가 지금보다 적지 않으면 통과. 통과한 방식이 여럿이면 위험 오답이
가장 적은 것, 같으면 더 단순한 것(나눔 < 나눔+A < 나눔+A+제목검색). 하나도 통과하지 못하면 지금을 유지하고 가짜정보 작업은 멈춘다.

사용(text-extraction 환경, scripts 폴더):
  python measure_final.py --wiki-index ..\\data\\wiki_index.sqlite3 --ollama-url http://localhost:11434 --ollama-model qwen3.5:4b --dump ..\\speaker_pipeline_dump.json --out ..\\final_dump.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import traceback
from pathlib import Path

import misinfo_infer
import misinfo_lib
from kiwipiepy import Kiwi
from measure_retrieval import CASES_DIR, subject_titles
from misinfo_timing import StepTimer
from nli_check import NliModel, default_model_path
from wiki_index import extract_keywords, title_chunks

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NAMES = ("지금", "나눔", "나눔+A", "나눔+A+제목검색")
TITLE_SLOTS = 2
_original_rerank = misinfo_infer.rerank


def title_aware_rerank(kiwi, conn):
    """검색 후보에 '문장 속 명사와 같은 제목의 문서'를 더하고, 그런 문서 최대 TITLE_SLOTS개에 상위 자리를 준다."""
    def rerank(text, candidates, top_k):
        keywords = extract_keywords(kiwi, text)
        seen = {c.rowid for c in candidates}
        pool = list(candidates)
        matched = set()
        for title in [t for t in subject_titles(kiwi, text) if len(t) >= 2][:6]:
            chunks = title_chunks(conn, title, keywords)
            if chunks:
                matched.add(title)
            for c in chunks:
                if c.rowid not in seen:
                    seen.add(c.rowid)
                    pool.append(c)
        if not pool:
            return []
        query_vec = misinfo_infer.embed(["query: " + text])
        passage_vecs = misinfo_infer.embed([f"passage: {c.title} {c.text}" for c in pool])
        order = [pool[i] for i in (passage_vecs @ query_vec.T).squeeze(-1).argsort(descending=True).tolist()]
        picked, counts = [], {}
        for c in order:  # 제목이 맞는 문서마다 가장 높은 조각 하나
            if c.title in matched and c.title not in counts and len(picked) < TITLE_SLOTS:
                picked.append(c)
                counts[c.title] = 1
        for c in order:  # 나머지는 지금처럼 e5 순서, 문서당 2개까지
            if len(picked) >= top_k:
                break
            if c not in picked and counts.get(c.title, 0) < misinfo_infer.MAX_CHUNKS_PER_TITLE:
                picked.append(c)
                counts[c.title] = counts.get(c.title, 0) + 1
        rank = {id(c): i for i, c in enumerate(order)}
        return sorted(picked, key=lambda c: rank[id(c)])
    return rerank


def run_current(kiwi, conn, sentence, args, nli, template, rerank=None) -> dict:
    """judge_sentence를 돌리고 화면 결과와 판정 입력 문서 제목을 남긴다."""
    seen = {}
    original_fit = misinfo_infer.fit_blocks

    def fit(blocks, *a, **kw):
        kept = original_fit(blocks, *a, **kw)
        seen["titles"] = [t for t, _ in kept]
        return kept

    misinfo_infer.fit_blocks = fit
    misinfo_infer.rerank = rerank or _original_rerank
    timer = StepTimer()
    try:
        shown = misinfo_infer.judge_sentence(kiwi, conn, sentence, args.ollama_url, args.ollama_model, 5,
                                             nli.contradiction_scores, timer=timer, judge_template=template)
    except Exception:
        shown, timer.outcome, seen["error"] = None, "error", traceback.format_exc(limit=2)
    finally:
        misinfo_infer.fit_blocks, misinfo_infer.rerank = original_fit, _original_rerank
    return {"shown": shown, "outcome": timer.outcome, "titles": seen.get("titles", []), "seconds": timer.seconds,
            **({"error": seen["error"]} if "error" in seen else {})}


def run_claim(kiwi, conn, sentence, claim, speaker, args, nli, rerank=None) -> dict:
    """분류기 방식: 정리된 내용으로 검색·판정(지시문 A) 후 화면 표시 여부."""
    misinfo_infer.rerank = rerank or _original_rerank
    timer = StepTimer()
    try:
        judged = misinfo_infer.judge_claim(kiwi, conn, claim, args.ollama_url, args.ollama_model, 5, False, timer,
                                           misinfo_lib.CLAIM_JUDGE_TEMPLATE_A)
        if judged is None:
            return {"shown": None, "outcome": timer.outcome, "titles": [], "seconds": timer.seconds}
        blocks, verdict = judged
        shown = misinfo_infer.show_if_wrong(sentence, claim, speaker, blocks, verdict, nli.contradiction_scores,
                                            nli.entailment_scores, timer)
        return {"shown": shown, "outcome": timer.outcome, "titles": [t for t, _ in blocks], "seconds": timer.seconds,
                "label": verdict["label"]}
    except Exception:
        return {"shown": None, "outcome": "error", "titles": [], "seconds": timer.seconds, "error": traceback.format_exc(limit=2)}
    finally:
        misinfo_infer.rerank = _original_rerank


def old_row(row: dict, kiwi, conn, args, nli, title_rerank) -> dict:
    """어제 기록이 있는 문장: 지금·나눔·나눔+A는 기록으로, 제목검색만 새로 돌린다."""
    v2 = row["분류기v2"]
    negated = v2.get("speaker") == misinfo_lib.SPEAKER_DENIES and bool(v2.get("claim"))
    out = {"sentence": row["sentence"], "expected": row["expected"], "file": row["file"], "negated": negated,
           "speaker": v2.get("speaker"), "claim": v2.get("claim")}
    claim_rec = {"shown": v2["shown"], "titles": [t for t, _ in v2.get("blocks", [])], "outcome": v2["outcome"]}
    now = {"shown": row["지금"]["shown"], "titles": [t for t, _ in row["지금"]["blocks"]], "outcome": row["지금"]["outcome"]}
    a = {"shown": row["A"]["shown"], "titles": [t for t, _ in row["A"]["blocks"]], "outcome": row["A"]["outcome"]}
    out["지금"] = now
    out["나눔"] = claim_rec if negated else now
    out["나눔+A"] = claim_rec if negated else a
    if negated:
        out["나눔+A+제목검색"] = run_claim(kiwi, conn, row["sentence"], v2["claim"], v2["speaker"], args, nli, title_rerank)
    else:
        out["나눔+A+제목검색"] = run_current(kiwi, conn, row["sentence"], args, nli, misinfo_lib.JUDGE_TEMPLATE_A, title_rerank)
    return out


def new_row(case: dict, kiwi, conn, args, nli, speaker_model, title_rerank) -> dict:
    """새 문장: 전부 새로 돌린다(서버 judge_sentence_speaker와 같은 정리·분류기 순서)."""
    sentence = case["sentence"]
    timer = StepTimer()
    claim = misinfo_infer.extract_claim(sentence, args.ollama_url, args.ollama_model, timer)
    speaker = misinfo_infer.speaker_of(speaker_model, sentence, claim) if claim else None
    invented = misinfo_infer.unsupported_words(kiwi, sentence, claim) if claim else []
    negated = speaker == misinfo_lib.SPEAKER_DENIES and not invented
    out = {"sentence": sentence, "expected": case.get("expected", ""), "file": case["file"], "negated": negated,
           "speaker": speaker, "claim": claim, "extract_seconds": timer.seconds}
    out["지금"] = run_current(kiwi, conn, sentence, args, nli, misinfo_lib.JUDGE_TEMPLATE)
    if negated:
        claim_rec = run_claim(kiwi, conn, sentence, claim, speaker, args, nli)
        out["나눔"] = out["나눔+A"] = claim_rec
        out["나눔+A+제목검색"] = run_claim(kiwi, conn, sentence, claim, speaker, args, nli, title_rerank)
    else:
        out["나눔"] = out["지금"]
        out["나눔+A"] = run_current(kiwi, conn, sentence, args, nli, misinfo_lib.JUDGE_TEMPLATE_A)
        out["나눔+A+제목검색"] = run_current(kiwi, conn, sentence, args, nli, misinfo_lib.JUDGE_TEMPLATE_A, title_rerank)
    return out


def tally(rows: list[dict], name: str) -> tuple[int, int, int]:
    false_all = sum(1 for r in rows if r["expected"] == "contradiction")
    det = sum(1 for r in rows if r["expected"] == "contradiction" and r[name]["shown"])
    risky = sum(1 for r in rows if r["expected"] != "contradiction" and r[name]["shown"])
    return det, false_all, risky


def summarize(rows: list[dict], gold: dict) -> None:
    new = [r for r in rows if r["file"] == "cases_fresh2.json"]
    old = [r for r in rows if r["file"] != "cases_fresh2.json"]
    print("\n===== [2/2] 요약 =====")
    for label, subset in (("새 문장(판단 기준)", new), ("기존 225문장(나눔 규칙을 정한 문장 - 참고)", old)):
        print(f"[{label}] 문장 {len(subset)}")
        for name in NAMES:
            det, false_all, risky = tally(subset, name)
            print(f"  {name}: 탐지 {det}/{false_all} | 위험 오답 {risky}")
    passed = []
    for name in NAMES[1:]:
        nd, _, nr = tally(new, name)
        od, _, orisk = tally(old, name)
        bd, _, br = tally(new, "지금")
        bod, _, bor = tally(old, "지금")
        ok = nr < br and nd >= bd and orisk <= bor and od >= bod
        print(f"판단 {name}: 새 문장 위험 {br}->{nr}, 탐지 {bd}->{nd} / 기존 위험 {bor}->{orisk}, 탐지 {bod}->{od} => {'통과' if ok else '불통과'}")
        if ok:
            passed.append((nr + orisk, NAMES.index(name), name))
    if passed:
        print(f"===== 최종: {min(passed)[2]} 채택 (통과한 방식: {', '.join(p[2] for p in sorted(passed))}) =====")
    else:
        print("===== 최종: 통과한 방식 없음 -> 지금(06ba7c5) 유지 =====")
    # 검색만 비교: 정답 문서표가 있는 문장에서 맞는 문서가 판정 입력에 들어갔는지.
    for name in ("지금", "나눔+A+제목검색"):
        hits = total = 0
        for r in old:
            g = gold.get(r["sentence"])
            if not g or g.get("gold_status") != "확정" or not g.get("gold_docs"):
                continue
            total += 1
            docs = set(g["gold_docs"]) | set(g.get("allowed_docs") or [])
            hits += bool(docs & set(r[name]["titles"]))
        print(f"맞는 문서가 판정 입력에 들어간 문장({name}): {hits}/{total}")
    for name in NAMES[1:]:
        changed = [r for r in rows if bool(r["지금"]["shown"]) != bool(r[name]["shown"])]
        print(f"----- 지금 -> {name}: 화면이 바뀐 문장 {len(changed)}개 -----")
        for r in changed:
            kind = ("탐지 생김" if r[name]["shown"] else "탐지 잃음") if r["expected"] == "contradiction" else (
                "위험 오답 생김" if r[name]["shown"] else "위험 오답 사라짐")
            print(f"  [{kind}] ({r['file']}) {r['sentence']} (분류기: {r.get('speaker')})")
            for side in ("지금", name):
                s = r[side]["shown"]
                if s:
                    ev = s["evidence"][0]
                    print(f"      {side} 이유: {s['reason'][:120]} | 근거: ({ev['title']}) {ev['text'][:100]}")
    errors = sum(1 for r in rows for n in NAMES if r[n].get("outcome") == "error")
    print(f"오류 {errors}건")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-index", required=True, type=Path)
    parser.add_argument("--ollama-url", required=True)
    parser.add_argument("--ollama-model", required=True)
    parser.add_argument("--dump", required=True, type=Path, help="measure_speaker_pipeline.py 기록(기존 225문장)")
    parser.add_argument("--out", required=True, type=Path, help="이번 측정 기록(문장마다 갱신)")
    parser.add_argument("--summary-only", action="store_true", help="측정 없이 --out 기록으로 요약만 다시 낸다")
    args = parser.parse_args()
    gold = {g["sentence"]: g for g in json.loads((CASES_DIR / "retrieval_gold.json").read_text(encoding="utf-8"))}
    if args.summary_only:
        summarize(json.loads(args.out.read_text(encoding="utf-8")), gold)
        return
    fresh_path = CASES_DIR / "cases_fresh2.json"
    speaker_dir = DATA_DIR / "speaker_model_v2"
    missing = [str(p) for p in (args.dump, fresh_path, speaker_dir) if not p.exists()]
    if missing:
        parser.error("없는 파일·폴더: " + ", ".join(missing))
    old_rows = json.loads(args.dump.read_text(encoding="utf-8"))
    fresh = [{**c, "file": "cases_fresh2.json"} for c in json.loads(fresh_path.read_text(encoding="utf-8"))]
    conn = sqlite3.connect(f"file:{args.wiki_index.resolve().as_posix()}?mode=ro", uri=True)
    kiwi = Kiwi()
    nli = NliModel(default_model_path())
    speaker_model = NliModel(str(speaker_dir), load_step="load_speaker")
    title_rerank = title_aware_rerank(kiwi, conn)
    total = len(fresh) + len(old_rows)
    print(f"새 문장 {len(fresh)}개 + 기존 {len(old_rows)}개 = {total}개")
    print("===== [1/2] 진행 =====")
    rows, start = [], time.perf_counter()
    try:
        for index, item in enumerate(fresh + old_rows, start=1):
            if index <= len(fresh):
                rows.append(new_row(item, kiwi, conn, args, nli, speaker_model, title_rerank))
            else:
                rows.append(old_row(item, kiwi, conn, args, nli, title_rerank))
            if index % 10 == 0 or index == total:
                elapsed = time.perf_counter() - start
                print(f"진행 {index}/{total} 문장, {elapsed:.0f}초 지남, 남은 시간 약 {elapsed / index * (total - index):.0f}초", flush=True)
            try:
                args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            except OSError as error:
                print(f"(기록 파일 쓰기 실패 - 측정은 계속: {error})", flush=True)
    finally:
        misinfo_infer.unload_ollama_model(args.ollama_url, args.ollama_model)
    summarize(rows, gold)
    print(f"전체 기록: {args.out}")


if __name__ == "__main__":
    main()
