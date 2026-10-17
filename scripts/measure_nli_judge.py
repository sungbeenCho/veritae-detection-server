"""4B 판정 없이 '분류기 + NLI'가 판정하는 구조를, measure_speaker_pipeline.py가 남긴 기록(dump)으로 계산해 지금(06ba7c5)과 비교한다.
4B는 다시 돌리지 않는다(e5·NLI만, 데스크탑 몇 분).

배경(2026-10-18 225문장 측정): 분류기는 부정·전언을 잘 읽었지만(오류 2), 분류기 방식에서 잃은 탐지·새 위험 오답 대부분이 4B
판정 단계에서 나왔다(지시문 한 줄에 18문장이 바뀔 만큼 흔들림). 그래서 4B에게서 판정과 이유 쓰기를 빼 본다:
  1) 정리(4B, 기록에 있는 것) -> 2) 분류기: 말한 사람이 그 내용을 맞다고/아니라고/사실 없음(기록에 있는 것)
  -> 3) 검색(기록의 판정 입력 문단 = 정리된 내용으로 찾은 문단) -> 4) NLI: e5로 내용과 가까운 문장 15개를 골라 모순·함의 점수
  -> 5) 맞다고 한 내용과 모순인 문장, 또는 아니라고 한 내용을 뒷받침하는 문장이 기준 이상이면 띄움
  -> 6) 이유는 정해진 문장틀: "위키백과 '<문서>' 문서에는 '<근거 문장>'라고 나와 있습니다."
과거에 같은 계열이 실패한 적이 있다: NLI가 혼자 근거를 고르면 주제만 같은 엉뚱한 문장에 높은 점수를 줬다(10-01, 지구본 받침대
베어링). 그래서 판단 기준을 측정 전에 정한다:
  - 기준 0.8(지금 서버의 NLI_THRESHOLD)로 판단한다. 다른 기준값은 참고로만 출력한다(같은 문장으로 기준을 고르면 결과가 부풀려진다).
  - 지금보다 위험 오답(틀리지 않은 문장을 띄움)이 줄고 맞는 탐지를 잃지 않아야 통과. 위험 오답이 지금(19)보다 많으면 이 구조는 접는다.

사용(text-extraction 환경, scripts 폴더):
  python measure_nli_judge.py --dump ..\\speaker_pipeline_dump.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DENIES = "아니라고 함"
STATES = "맞다고 함"
PRIMARY = 0.8
EXTRA_THRESHOLDS = (0.9, 0.95, 0.99)


def reason_text(title: str, sentence: str) -> str:
    return f"위키백과 '{title}' 문서에는 '{sentence}'라고 나와 있습니다."


def score_row(r: dict, nli, shortlist, candidate_sentences) -> dict | None:
    """분류기 방식 기록 하나에서 NLI 판정에 필요한 점수. 판정 전에 끝난 문장(정리 빈 내용, 사실 없음, 지어냄, 검색 없음)은 None."""
    if r.get("speaker") not in (STATES, DENIES) or not r.get("blocks") or not r.get("claim"):
        return None
    blocks = [tuple(b) for b in r["blocks"]]
    candidates = shortlist(r["claim"], candidate_sentences(blocks))
    if not candidates:
        return None
    probs = nli.label_probabilities([s for _, s in candidates], r["claim"])
    key = "contradiction" if r["speaker"] == STATES else "entailment"
    scored = sorted(((p[key], t, s) for p, (t, s) in zip(probs, candidates)), reverse=True)
    best, title, sentence = scored[0]
    # 같은 후보에서 반대쪽 최고점도 남긴다 - 띄운 문장이 엉뚱한지 사람이 볼 때 참고한다.
    other = "entailment" if key == "contradiction" else "contradiction"
    return {"speaker": r["speaker"], "claim": r["claim"], "score": best, "title": title, "sentence": sentence,
            "other_best": max(p[other] for p in probs)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--nli-model", default=None)
    parser.add_argument("--out", type=Path, help="문장별 결과 JSON(사람 채점용)")
    args = parser.parse_args()

    from misinfo_infer import shortlist_sentences
    from nli_check import NliModel, candidate_sentences, default_model_path

    rows = json.loads(args.dump.read_text(encoding="utf-8"))
    nli = NliModel(args.nli_model or default_model_path())
    names = ("분류기v2", "분류기v1")
    results = []
    import time
    start = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        entry = {"sentence": row["sentence"], "expected": row["expected"], "file": row["file"], "now_shown": bool(row["지금"]["shown"]),
                 "now": row["지금"]["shown"]}
        for name in names:
            entry[name] = score_row(row[name], nli, shortlist_sentences, candidate_sentences)
        results.append(entry)
        # 끝까지 아무것도 안 찍으면 멈춘 것처럼 보인다 - 10문장마다 진행 상황과 남은 시간을 찍는다.
        if index % 10 == 0 or index == len(rows):
            elapsed = time.perf_counter() - start
            print(f"진행 {index}/{len(rows)} 문장, {elapsed:.0f}초 지남, 남은 시간 약 {elapsed / index * (len(rows) - index):.0f}초", flush=True)

    def shown(e: dict, name: str, threshold: float) -> bool:
        s = e[name]
        return s is not None and s["score"] >= threshold

    groups = {"전체": results, "기존 회귀 사례": [e for e in results if e["file"] not in ("cases_fresh.json", "cases_false_negation.json")],
              "새 문장(cases_fresh)": [e for e in results if e["file"] == "cases_fresh.json"],
              "참인 내용을 아니라고 함": [e for e in results if e["file"] == "cases_false_negation.json"]}
    print(f"===== 요약 (판단 기준 {PRIMARY}, 다른 기준값은 참고) =====")
    for group, subset in groups.items():
        false_all = sum(1 for e in subset if e["expected"] == "contradiction")
        now_det = sum(1 for e in subset if e["expected"] == "contradiction" and e["now_shown"])
        now_risky = sum(1 for e in subset if e["expected"] != "contradiction" and e["now_shown"])
        print(f"[{group}] 문장 {len(subset)} | 지금: 탐지 {now_det}/{false_all}, 위험 오답 {now_risky}")
        for name in names:
            for threshold in (PRIMARY,) + EXTRA_THRESHOLDS:
                det = sum(1 for e in subset if e["expected"] == "contradiction" and shown(e, name, threshold))
                risky = sum(1 for e in subset if e["expected"] != "contradiction" and shown(e, name, threshold))
                mark = "" if threshold == PRIMARY else " (참고)"
                print(f"  NLI판정+{name} 기준 {threshold}{mark}: 탐지 {det}/{false_all}, 위험 오답 {risky}")
    print()

    name = "분류기v2"
    now_risky = sum(1 for e in results if e["expected"] != "contradiction" and e["now_shown"])
    new_risky = sum(1 for e in results if e["expected"] != "contradiction" and shown(e, name, PRIMARY))
    lost = [e for e in results if e["expected"] == "contradiction" and e["now_shown"] and not shown(e, name, PRIMARY)]
    verdict = ("접음(위험 오답이 지금보다 많음)" if new_risky > now_risky else
               "통과(위험 오답 줄고 탐지 잃음 0)" if new_risky < now_risky and not lost else
               "사람 채점 필요(잃은 탐지가 나쁜 탐지뿐인지 확인)" if new_risky < now_risky else "불통과(위험 오답이 줄지 않음)")
    print(f"===== 판단(NLI판정+{name}, 기준 {PRIMARY}): 위험 오답 {now_risky} -> {new_risky}, 잃은 탐지 {len(lost)} => {verdict} =====")
    print()

    def line(e: dict) -> str:
        s = e[name]
        if s is None:
            return "판정 전에 끝남(정리 빈 내용·사실 없음·지어냄·검색 없음)"
        text = f"분류기 {s['speaker']} | 내용: {s['claim']} | 점수 {s['score']:.2f}(반대쪽 최고 {s['other_best']:.2f})"
        if s["score"] >= PRIMARY:
            text += f" | 화면 이유: {reason_text(s['title'], s['sentence'])}"
        else:
            text += f" | 가장 높은 문장: ({s['title']}) {s['sentence'][:100]}"
        return text

    print(f"===== 위험 오답(NLI판정+{name}) - 틀리지 않은 문장을 띄움 =====")
    for e in results:
        if e["expected"] != "contradiction" and shown(e, name, PRIMARY):
            print(f"  [{e['expected']}] ({e['file']}) {e['sentence']}{' (지금도 위험 오답)' if e['now_shown'] else ''}")
            print(f"      {line(e)}")
    print(f"===== 잃은 탐지(지금은 띄우는데 NLI판정은 안 띄움) {len(lost)}개 =====")
    for e in lost:
        now = e["now"]["evidence"][0]
        print(f"  ({e['file']}) {e['sentence']}")
        print(f"      지금 이유: {e['now']['reason'][:120]} | 지금 근거: ({now['title']}) {now['text'][:100]}")
        print(f"      {line(e)}")
    gained = [e for e in results if e["expected"] == "contradiction" and not e["now_shown"] and shown(e, name, PRIMARY)]
    print(f"===== 새로 잡은 탐지 {len(gained)}개 =====")
    for e in gained:
        print(f"  ({e['file']}) {e['sentence']}")
        print(f"      {line(e)}")
    kept = [e for e in results if e["expected"] == "contradiction" and e["now_shown"] and shown(e, name, PRIMARY)]
    print(f"===== 둘 다 잡은 탐지 {len(kept)}개 - 화면 이유 비교 =====")
    for e in kept:
        print(f"  ({e['file']}) {e['sentence']}")
        print(f"      지금 이유: {e['now']['reason'][:150]}")
        print(f"      NLI 이유: {reason_text(e[name]['title'], e[name]['sentence'])[:200]}")
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"전체 기록: {args.out}")


if __name__ == "__main__":
    main()
