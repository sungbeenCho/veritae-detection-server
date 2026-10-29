"""measure_speaker_pipeline.py가 남긴 기록(dump)으로, 분류기 방식에서 화면 결과가 지금(06ba7c5)과 달라진 문장마다 어느 단계에서
갈렸는지 나눈다. 4B를 다시 돌리지 않는다(몇 초).

배경(2026-10-18 225문장 측정): 분류기v2가 새 문장의 부정형 위험 오답을 10 -> 4로 줄였지만 맞는 탐지를 약 11개 잃고 대상 불명
문장 위험 오답을 새로 만들었다. 분류기 방식은 정리·분류기·정리된 내용으로 검색·판정 지시문을 한꺼번에 바꿨기 때문에 원인이 섞여
있다. 단계를 이렇게 나눈다(앞 단계부터 본다):
  - 정리: 4B 정리가 빈 내용(따질 사실 없음)이거나 원문에 없는 말(extract_unfaithful)이라 판정 전에 끝남.
  - 분류기: 분류기가 사실 없음이라 끝남, 또는 두 분류기의 판단이 다름.
  - 검색: 지금이 인용한 문서가 분류기 방식의 판정 입력 문단에 없음.
  - 판정: 같은 문서가 들어왔는데 판정이 다름. 지시문 A만 바꾼 방식(A)도 같은 화면 결과면 '지시문 A 영향'으로 따로 표시한다.
  - 표시 조건: 판정은 띄울 쪽인데 인용 문단·이유가 비어 안 띄움.

사용(text-extraction 환경, scripts 폴더):
  python analyze_speaker_dump.py --dump ..\\speaker_pipeline_dump.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DENIES = "아니라고 함"
NO_FACT = "사실 없음"


def wants_show(r: dict) -> bool:
    """판정·분류기만 보면 띄울 쪽인지(표시 조건 이전)."""
    return (r.get("speaker") == DENIES and r.get("label") == "지지") or (r.get("speaker") not in (DENIES, NO_FACT, None)
                                                                        and r.get("label") == "반박")


def stage(row: dict, name: str) -> str:
    now, new = row["지금"], row[name]
    if new["outcome"] in ("no_fact", "extract_unfaithful", "extract_format_error"):
        return "정리"
    if new["outcome"] == "no_fact_speaker":
        return "분류기(사실 없음)"
    if bool(now["shown"]) and not new["shown"] and wants_show(new):
        return "표시 조건"
    # 판정은 그대로 두고 분류기의 맞다고/아니라고만 반대였다면 화면이 지금과 같았을 경우 - 분류기 판단이 갈랐다.
    flipped = {**new, "speaker": DENIES if new.get("speaker") == "맞다고 함" else "맞다고 함"}
    if new.get("speaker") in ("맞다고 함", DENIES) and "label" in new and wants_show(flipped) == bool(now["shown"]):
        return "분류기(맞다고/아니라고)"
    if "blocks" not in new:
        return "검색(입력 문단 없음)" if new["outcome"] == "no_evidence_found" else f"기타({new['outcome']})"
    new_titles = {t for t, _ in new["blocks"]}
    if now.get("cited") and not set(now["cited"]) & new_titles:
        return "검색(지금이 인용한 문서 없음)"
    return "판정"


def short(r: dict) -> str:
    text = f"결과={r['outcome']}"
    if r.get("claim") is not None:
        text += f" | 정리: {r['claim']}"
    if r.get("speaker"):
        text += f" | 분류기: {r['speaker']}"
    if "label" in r:
        text += f" | 판정: {r['label']} | 인용: {', '.join(r.get('cited', [])) or '(없음)'} | 이유: {r.get('reason', '')[:120]}"
    if r.get("blocks"):
        text += f" | 입력 문서: {', '.join(dict.fromkeys(t for t, _ in r['blocks']))}"
    return text


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, type=Path)
    args = parser.parse_args()
    rows = json.loads(args.dump.read_text(encoding="utf-8"))

    for name in ("분류기v2", "분류기v1"):
        changed = [row for row in rows if bool(row["지금"]["shown"]) != bool(row[name]["shown"])]
        print(f"===== 지금 -> {name}: 화면이 바뀐 문장 {len(changed)}개 =====")
        tally: Counter = Counter()
        for row in changed:
            lost_or_gained = ("탐지 잃음" if row["지금"]["shown"] else "탐지 생김") if row["expected"] == "contradiction" else (
                "위험 오답 사라짐" if row["지금"]["shown"] else "위험 오답 생김")
            where = stage(row, name)
            same_as_a = bool(row["A"]["shown"]) == bool(row[name]["shown"])
            tally[(lost_or_gained, where + (" + 지시문 A도 같음" if same_as_a else ""))] += 1
            if name == "분류기v2":
                print(f"[{lost_or_gained}] [{where}{' / 지시문 A도 같음' if same_as_a else ''}] ({row['file']}) {row['sentence']}")
                print(f"    지금: {short(row['지금'])}")
                print(f"    {name}: {short(row[name])}")
                if row["분류기v1"].get("speaker") != row["분류기v2"].get("speaker"):
                    print(f"    (v1 분류기는 {row['분류기v1'].get('speaker')})")
        print(f"----- {name} 단계별 집계 -----")
        for (kind, where), count in sorted(tally.items()):
            print(f"  {kind} | {where}: {count}")
        print()

    # 판정 지시문 A만 바꾼 효과(검색·입력은 지금과 같다) - 분류기 방식의 변화 중 지시문 A 몫을 가늠한다.
    changed_a = [row for row in rows if bool(row["지금"]["shown"]) != bool(row["A"]["shown"])]
    print(f"===== 참고: 지금 -> A(지시문만 바꿈) 화면이 바뀐 문장 {len(changed_a)}개 =====")
    for row in changed_a:
        print(f"  [{row['expected']}] {row['sentence']} : 지금 {row['지금'].get('label', row['지금']['outcome'])} -> A {row['A'].get('label', row['A']['outcome'])}")


if __name__ == "__main__":
    main()
