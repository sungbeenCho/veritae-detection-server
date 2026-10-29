"""'NLI 먼저' 구조가 거짓 문장을 얼마나 놓치는지 재는 도구. 데스크탑에서 한 번 돌린다(LLM/Ollama 불필요).

배경(2026-10-13): 가짜정보 판정은 모든 문장에 LLM을 먼저 돌려 문장당 9~19초가 든다. 모든 문장을 위키 검색 + NLI(빠른
모델)로 먼저 검사하고, 거짓 후보가 나온 문장에만 LLM을 돌리면 대부분의 문장에서 LLM 호출이 사라진다. 다만 이 구조에서는
LLM이 정리해 주던 핵심 주장 없이 NLI에 문장 원문을 넣게 된다 - 옛 NLI는 "~라는 이야기를 들었어요"의 정답 근거에
0.058점을 줘서 반박을 놓쳤다. 새 NLI 모델로 원문을 넣었을 때 놓치는 비율이 얼마인지 숫자를 먼저 본다.

검색은 서버와 같은 방식(키워드 검색 -> e5 재정렬 -> 앞뒤 문맥과 첫 문단 -> 문장 나누기)이고, 후보 문장 전부를 NLI로
점수 매긴 뒤 최고점으로 문장을 통과/탈락시킨다고 가정해 임계값별로 센다.

사용(text-extraction 환경, C:\\ai\\veritae-detection-server\\scripts):
  python measure_nli_gate.py --wiki-index ..\\data\\wiki_index.sqlite3 --dump ..\\nli_gate.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import misinfo_infer
from kiwipiepy import Kiwi
from misinfo_lib import fit_blocks
from nli_check import NliModel, candidate_sentences, default_model_path
from wiki_index import expand_with_neighbors, extract_keywords, search

CASE_FILES = ["cases_e2e.json", "cases_grounding.json", "cases_heldout.json"]
CASES_DIR = Path(__file__).resolve().parent.parent / "tests" / "regression"
DEFAULT_THRESHOLDS = [0.3, 0.5, 0.7, 0.8, 0.9]


def gate_scores(kiwi, conn, sentence: str, evidence_count: int, nli) -> dict:
    """서버와 같은 검색으로 후보 문장을 모아 문장 원문을 가설로 NLI 점수를 매긴다."""
    keywords = extract_keywords(kiwi, sentence)
    chunks = search(conn, keywords, limit=50) if keywords else []
    empty = {"max_score": 0.0, "best_title": None, "best_sentence": None, "candidate_count": 0, "candidates": []}
    if not chunks:
        return empty
    top_chunks = misinfo_infer.rerank(sentence, chunks, top_k=evidence_count)
    candidates = candidate_sentences(fit_blocks(expand_with_neighbors(conn, top_chunks)))
    if not candidates:
        return empty
    scores = nli([s for _, s in candidates], sentence)
    scored = [(title, text, float(score)) for (title, text), score in zip(candidates, scores)]
    best_title, best_sentence, best = max(scored, key=lambda item: item[2])
    return {
        "max_score": best, "best_title": best_title, "best_sentence": best_sentence,
        "candidate_count": len(scored), "candidates": [list(item) for item in scored],
    }


def summarize(records: list[dict], thresholds: list[float]) -> dict[float, dict]:
    """임계값마다: 거짓 문장 중 통과한 비율(recall), 놓친 문장, 거짓이 아닌데 통과한 건수, LLM을 건너뛰는 문장 비율."""
    summary: dict[float, dict] = {}
    false_cases = [r for r in records if r["expected"] == "contradiction"]
    for threshold in thresholds:
        passed_false = [r for r in false_cases if r["max_score"] >= threshold]
        missed = [r for r in false_cases if r["max_score"] < threshold]
        by_category: dict[str, list[bool]] = {}
        for r in false_cases:
            by_category.setdefault(r["category"], []).append(r["max_score"] >= threshold)
        skipped = [r for r in records if r["max_score"] < threshold]
        summary[threshold] = {
            "recall": len(passed_false) / len(false_cases) if false_cases else None,
            "missed": len(missed),
            "missed_sentences": [r["sentence"] for r in missed],
            "false_gate_pass": sum(1 for r in records if r["expected"] != "contradiction" and r["max_score"] >= threshold),
            "llm_skipped": len(skipped),
            "llm_skipped_rate": len(skipped) / len(records) if records else None,
            "recall_by_category": {c: sum(v) / len(v) for c, v in by_category.items()},
        }
    return summary


def format_report(records: list[dict], summary: dict[float, dict], seconds_per_case: float) -> str:
    false_count = sum(1 for r in records if r["expected"] == "contradiction")
    lines = [f"문장 {len(records)}개(거짓 {false_count}개), 문장당 검색+NLI 평균 {seconds_per_case:.2f}초", ""]
    lines.append("임계값 | 거짓 통과(recall) | 놓침 | 거짓 아닌데 통과(LLM이 거름) | LLM 건너뜀")
    for threshold, s in summary.items():
        recall = "-" if s["recall"] is None else f"{s['recall']:.0%}"
        skipped = "-" if s["llm_skipped_rate"] is None else f"{s['llm_skipped']}개({s['llm_skipped_rate']:.0%})"
        lines.append(f"{threshold:.1f}   | {recall:>6}            | {s['missed']:>3}  | {s['false_gate_pass']:>3}                         | {skipped}")
    lines += ["", "분류별 거짓 통과율(임계값 0.8):"]
    reference = summary.get(0.8) or next(iter(summary.values()))
    for category, rate in sorted(reference["recall_by_category"].items()):
        lines.append(f"  {category}: {rate:.0%}")
    lines += ["", "거짓 문장별 최고 NLI 점수(낮은 순) - 최고점 문장을 직접 읽어 맞는 근거인지 확인:"]
    for r in sorted((r for r in records if r["expected"] == "contradiction"), key=lambda r: r["max_score"]):
        lines.append(f"  {r['max_score']:.3f} [{r['category']}] {r['sentence']}")
        lines.append(f"        -> ({r.get('best_title')}) {r.get('best_sentence')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-index", required=True, type=Path)
    parser.add_argument("--nli-model", default=default_model_path())
    parser.add_argument("--evidence-count", type=int, default=5)
    parser.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    parser.add_argument("--dump", type=Path, help="문장별 후보 문장 전체와 NLI 점수를 저장(노트북에서 임계값을 바꿔 가며 다시 계산용)")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    conn = sqlite3.connect(f"file:{args.wiki_index.resolve().as_posix()}?mode=ro", uri=True)
    kiwi = Kiwi()
    nli = NliModel(args.nli_model).contradiction_scores

    records: list[dict] = []
    start = time.perf_counter()
    for file in CASE_FILES:
        for case in json.loads((CASES_DIR / file).read_text(encoding="utf-8")):
            result = gate_scores(kiwi, conn, case["sentence"], args.evidence_count, nli)
            records.append({"expected": case["expected"], "category": case["category"], "sentence": case["sentence"], **result})
    seconds_per_case = (time.perf_counter() - start) / len(records)

    print(format_report(records, summarize(records, thresholds), seconds_per_case))
    if args.dump:
        args.dump.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n문장별 기록 {len(records)}건 저장: {args.dump}")


if __name__ == "__main__":
    main()
