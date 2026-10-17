"""화면 근거 문장을 고르는 범위 두 가지를 비교하는 측정 도구. measure_judge_prompt.py가 남긴 기록(dump)으로 근거 고르기만
다시 계산한다 - 4B는 다시 돌리지 않는다. 데스크탑에서 한 번 돌린다(e5·NLI 필요, 수 분).

배경(2026-10-17): 화면에는 이유 + 근거 문장 1개가 나가는데, 근거 문장은 판정이 인용한 문단과 상관없이 검색된 문단 전체에서
고른다(2026-10-01 결정 - 판정이 엉뚱한 문단을 인용해 맞는 근거 "학계의 중론은…"을 놓쳐서). 판정 지시문 비교(158문장) 사람 채점에서
화면 근거가 엉뚱한 경우의 다수가 인용하지 않은 문서의 문장이었다(김종완 가습기 일화, 달 착륙선 정의, 하춘화 등). 판정 지시문 A
(가장 직접적인 문단 1개)는 인용을 제대로 하게 됐으므로, 근거도 인용한 문단 안에서 고르면 이유와 근거가 같은 내용을 가리킨다.
위험: 10-01처럼 인용이 틀리면 맞는 근거를 놓친다 - 그래서 지금 지시문과 A 각각에서 두 범위를 비교한다.

  - 전체: 지금 서버 방식(검색된 문단 전체 → e5로 주장과 가까운 15개 → NLI 0.8 이상 중 이유와 가장 가까운 문장, 없으면 인용 문단에서
    이유와 가장 가까운 문장). 같은 스크립트로 다시 계산해 dump의 화면 근거와 같은지도 본다(대조군 - 다르면 e5·NLI 계산 오차).
  - 인용: 같은 절차를 판정이 인용한 문단 안에서만.
"전체" 근거가 NLI로 확인된 문장(nli)이었는데 "인용"에서 이유와 가까운 문장(reason_match)으로 떨어지면, 맞는 근거를 잃었을 수 있는
경우다 - 사람이 확인한다.

사용(text-extraction 환경, scripts 폴더):
  python measure_evidence_scope.py --dump ..\\judge_prompt_dump.json --out ..\\evidence_scope.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from misinfo_infer import closest_to_reason, shortlist_sentences
from nli_check import NLI_THRESHOLD, NliModel, candidate_sentences, default_model_path, pick_best_contradiction


def pick(nli, claim: str, reason: str, blocks: list[tuple[str, str]], cited: list[tuple[str, str]]) -> dict:
    """서버 judge_sentence의 근거 고르기와 같은 절차. blocks는 NLI 후보를 뽑을 문단, cited는 NLI가 못 찾았을 때 쓸 인용 문단."""
    candidates = shortlist_sentences(claim, candidate_sentences(blocks))
    if not candidates:
        return {"evidence": None, "by": "후보 없음", "top": None}
    best, scored = pick_best_contradiction(nli, candidates, claim)
    top = max(c["score"] for c in scored) if scored else None
    if best is not None:
        passing = [(c["title"], c["sentence"]) for c in scored if c["score"] >= NLI_THRESHOLD]
        return {"evidence": list(closest_to_reason(reason, passing)), "by": "nli", "top": top}
    fallback = candidate_sentences(cited) or candidate_sentences(blocks)
    if not fallback:
        return {"evidence": None, "by": "후보 없음", "top": top}
    return {"evidence": list(closest_to_reason(reason, fallback)), "by": "reason_match", "top": top}


def show(label: str, result: dict, cited_text: str) -> str:
    ev = result["evidence"]
    if ev is None:
        return f"    {label}: 없음 [{result['by']}]"
    top = f" nli최고={result['top']:.2f}" if result["top"] is not None else ""
    outside = "" if ev[1] in cited_text else "  <- 인용 밖 문단"
    return f"    {label}: ({ev[0]}) {ev[1]}  [{result['by']}{top}]{outside}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, type=Path, help="measure_judge_prompt.py의 --dump 파일")
    parser.add_argument("--nli-model", default=default_model_path())
    parser.add_argument("--out", type=Path, help="결과 JSON(사람 채점용)")
    args = parser.parse_args()
    rows = json.loads(args.dump.read_text(encoding="utf-8"))
    nli = NliModel(args.nli_model).contradiction_scores

    print("===== [1/2] 화면에 나간 문장별 근거 비교 =====")
    results = []
    for index, row in enumerate(rows, start=1):
        for name in ("지금", "A"):
            r = row.get(name) or {}
            if not r.get("shown"):
                continue
            blocks = [tuple(b) for b in r["blocks"]]
            cited = [blocks[i - 1] for i in r["evidence_ids"]]
            cited_text = "\n".join(b for _, b in cited)
            dumped = [r["shown"]["evidence"][0]["title"], r["shown"]["evidence"][0]["text"]]
            full = pick(nli, r["core_claim"], r["reason"], blocks, cited)
            scoped = pick(nli, r["core_claim"], r["reason"], cited, cited)
            entry = {"index": index, "prompt": name, "sentence": row["sentence"], "expected": row["expected"],
                     "reason": r["reason"], "cited": [t for t, _ in cited], "dumped": dumped, "full": full, "scoped": scoped,
                     "control_same": full["evidence"] == dumped, "same": scoped["evidence"] == full["evidence"],
                     "full_outside": full["evidence"] is not None and full["evidence"][1] not in cited_text,
                     "lost_nli": full["by"] == "nli" and scoped["by"] != "nli"}
            results.append(entry)
            mark = "같음" if entry["same"] else "바뀜"
            control = "" if entry["control_same"] else " | 대조군 불일치(서버 기록과 다시 계산한 전체가 다름)"
            print(f"#{index} [{row['expected']}] {name} | {mark}{control} | {row['sentence']}")
            print(f"    이유: {r['reason']}")
            print(show("전체", full, cited_text))
            if not entry["same"]:
                print(show("인용", scoped, cited_text))
    print()
    print("===== [2/2] 요약 =====")
    for name in ("지금", "A"):
        res = [x for x in results if x["prompt"] == name]
        print(f"{name}: 화면에 나간 {len(res)} | 대조군 불일치 {sum(not x['control_same'] for x in res)} | "
              f"전체 방식 근거가 인용 밖 문단 {sum(x['full_outside'] for x in res)} | 근거가 바뀐 {sum(not x['same'] for x in res)} | "
              f"NLI 확인 근거가 인용 범위에서 사라짐 {sum(x['lost_nli'] for x in res)} | "
              f"인용 범위에서 근거 못 찾음 {sum(x['scoped']['evidence'] is None for x in res)}")
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"전체 기록: {args.out}")


if __name__ == "__main__":
    main()
