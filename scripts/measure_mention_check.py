"""'이미 맞는 말을 하는 문장' 확인 방식을 정하는 측정 도구. 데스크탑에서 한 번 돌린다(LLM/Ollama/위키 인덱스 불필요, 1분 이내).

배경(2026-10-14): LLM이 반박이라고 맞게 판정해도, NLI가 위키에서 "주장과 모순"인 문장을 0.8점 이상으로 못 찾으면
판정을 버렸다. 위키가 "선풍기 사망설이란 … 사망할 수 있다는 미신이다"처럼 쓰면 NLI가 0.06점을 줘서 맞는 판정이
버려졌다(이미지 실측). 그래서 NLI가 판정을 버리지 못하게 바꾸고, LLM이 실제로 틀렸던 유형만 막는다 - "한국에는
선풍기를 틀고 자면 죽는다는 속설이 있다", "만리장성이 … 잘못 알려진 것이다"처럼 문장이 이미 맞는 말을 하는데
LLM이 반박이라고 착각한 경우(59문장 기록 2건씩, 두 기록).

막는 방식은 "문장(또는 LLM이 정리한 핵심 주장)과 LLM의 반박 이유가 같은 말인가"를 NLI로 보는 것이다. 어느 쪽을
전제로 둘지(방향)와 어떤 점수를 쓸지에 따라 결과가 달라서, 실제 LLM 판정 기록으로 전부 재 본다.
측정 데이터(tests/regression/mention_check_cases.json):
- record_*: 데스크탑 회귀 기록 두 개(candidates_e0134e3/6b819fb)에서 LLM이 반박이라고 한 문장 전부와 그때의 이유
- server_2026-10-13: 이번 이미지 실측(OCR 오타 문장, 핵심 주장이 "~속설이 있다"로 잘못 정리된 경우)
- paired_same_topic: 맞는 말을 하는 문장에 같은 주제의 실제 반박 이유를 붙인 것(LLM이 착각하면 나올 이유)
- ocr_typo_hearsay: 전해 들은 말에 EasyOCR이 실제로 낸 오타(를→틀, 었→없)를 넣은 것
expected가 show면 표시돼야 하고(막으면 안 됨), block이면 막아야 한다.

사용(text-extraction 환경, C:\\ai\\veritae-detection-server\\scripts):
  python measure_mention_check.py --dump ..\\mention_check.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from nli_check import NliModel, default_model_path

CASES_PATH = Path(__file__).resolve().parent.parent / "tests" / "regression" / "mention_check_cases.json"

# 방향 이름 -> (전제로 쓸 필드, 가설로 쓸 필드)
DIRECTIONS = {
    "이유→문장": ("reason", "sentence"),
    "이유→주장": ("reason", "core_claim"),
    "문장→이유": ("sentence", "reason"),
    "주장→이유": ("core_claim", "reason"),
}
# 점수가 이 값 이상이면 "같은 말"로 보고 막는다.
ENTAILMENT_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]

ProbabilityScorer = Callable[[list[str], str], list[dict[str, float]]]


def score_cases(cases: list[dict], probs: ProbabilityScorer) -> list[dict]:
    records = []
    for case in cases:
        scores = {}
        for direction, (premise_field, hypothesis_field) in DIRECTIONS.items():
            scores[direction] = probs([case[premise_field]], case[hypothesis_field])[0]
        records.append({**case, "scores": scores})
    return records


def evaluate_rule(records: list[dict], direction: str, label: str, threshold: float) -> dict:
    """records[*].scores[direction][label] >= threshold이면 막는다고 했을 때의 결과."""
    blocked = [r for r in records if r["scores"][direction][label] >= threshold]
    to_block = [r for r in records if r["expected"] == "block"]
    to_show = [r for r in records if r["expected"] == "show"]
    return {
        "direction": direction, "label": label, "threshold": threshold,
        "blocked_ok": sum(1 for r in blocked if r["expected"] == "block"),
        "block_total": len(to_block),
        "missed_blocks": [r["sentence"] for r in to_block if r not in blocked],
        "wrongly_blocked": [r["sentence"] for r in blocked if r["expected"] == "show"],
        "show_total": len(to_show),
    }


def summarize(records: list[dict]) -> list[dict]:
    """모든 (방향, 임계값) 조합. 표시돼야 할 문장을 덜 막는 순, 그다음 맞는 말을 더 많이 막는 순."""
    rows = [
        evaluate_rule(records, direction, "entailment", threshold)
        for direction in DIRECTIONS for threshold in ENTAILMENT_THRESHOLDS
    ]
    return sorted(rows, key=lambda r: (len(r["wrongly_blocked"]), -r["blocked_ok"]))


def best_rule(rows: list[dict]) -> dict | None:
    """표시돼야 할 문장은 하나도 막지 않으면서 맞는 말을 가장 많이 막는 조합. 그런 조합이 없으면 None."""
    safe = [r for r in rows if not r["wrongly_blocked"] and r["blocked_ok"] > 0]
    return max(safe, key=lambda r: r["blocked_ok"]) if safe else None


def format_report(records: list[dict], rows: list[dict]) -> str:
    lines = [
        f"측정 {len(records)}건: 표시돼야 함 {sum(r['expected'] == 'show' for r in records)}건, "
        f"막아야 함 {sum(r['expected'] == 'block' for r in records)}건",
        "",
        "방향 | 함의 점수 기준 | 막아야 할 것 중 막음 | 표시돼야 하는데 막음",
    ]
    for r in rows:
        lines.append(f"{r['direction']} | {r['threshold']:.1f} 이상 | {r['blocked_ok']}/{r['block_total']} | "
                     f"{len(r['wrongly_blocked'])}/{r['show_total']}")
    best = best_rule(rows)
    if best is None:
        lines += ["", "거짓 문장을 하나도 막지 않으면서 맞는 말을 막는 조합이 없다."]
    else:
        lines += ["", f"가장 나은 조합({best['direction']}, 함의 {best['threshold']:.1f} 이상)에서:"]
        lines += [f"  못 막은 맞는 말: {s}" for s in best["missed_blocks"]] or ["  못 막은 맞는 말: 없음"]
    lines += ["", "문장별 함의 점수(방향마다):"]
    for rec in sorted(records, key=lambda r: r["expected"]):
        values = " ".join(f"{d}={rec['scores'][d]['entailment']:.2f}" for d in DIRECTIONS)
        lines.append(f"  [{rec['expected']}] {values} | {rec['sentence']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--nli-model", default=default_model_path())
    parser.add_argument("--dump", type=Path, help="문장별 세 라벨 점수 전체를 저장(노트북에서 다른 기준을 다시 계산용)")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    records = score_cases(cases, NliModel(args.nli_model).label_probabilities)
    print(format_report(records, summarize(records)))
    if args.dump:
        args.dump.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n문장별 기록 {len(records)}건 저장: {args.dump}")


if __name__ == "__main__":
    main()
