"""말한 사람 판별 모델(train_nli.py --extra-data로 학습)을 평가 세트로 채점한다. 데스크탑에서 학습 뒤 한 번 돌린다.

문장을 말한 사람이 내용(claim)을 맞다고 하는지/아니라고 하는지/따질 사실이 없는지 맞히는지 본다. 가장 중요한 것은
맞다고 함 <-> 아니라고 함이 뒤바뀐 건수다 - 뒤바뀌면 맞는 말을 틀렸다고 띄우거나 틀린 말을 놓친다(사실 없음과 헷갈리면
판단불가로 물러날 뿐이다). 평가 세트는 tests/speaker/speaker_eval.json:
  - source 4B_run1/4B_run3: 4B가 실제로 정리한 내용(claim)에 사람이 정답을 다시 매긴 것 - 서비스에서 받을 입력과 같다.
  - subset "4B정리흠": 4B 정리가 "만" 누락·오타 남음·시제 바뀜처럼 흠이 있는 것 - 본 점수와 따로 센다.
  - source writer2: 학습 데이터와 다른 사람이 학습에 없는 표현(낭설·도시전설·팩트 아님 등)으로 쓴 것.
학습 데이터와 원문이 겹치지 않는다(2026-10-17 확인).

사용(text-extraction 환경, scripts 폴더):
  python eval_speaker.py --model ..\\data\\speaker_model --eval ..\\tests\\speaker\\speaker_eval.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# train_nli.SPEAKER_LABELS와 같은 대응.
SPEAKER_TO_NLI = {"맞다고 함": "entailment", "아니라고 함": "contradiction", "사실 없음": "neutral"}
NLI_TO_SPEAKER = {v: k for k, v in SPEAKER_TO_NLI.items()}


def predict(model_path: Path, rows: list[dict]) -> list[str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    id2label = model.config.id2label
    out: list[str] = []
    for start in range(0, len(rows), 32):
        batch = rows[start:start + 32]
        # 학습(train_nli.py)과 같게 token_type_ids를 빼고 128 토큰으로 자른다.
        inputs = tokenizer([r["sentence"] for r in batch], [r["claim"] for r in batch], padding=True, truncation=True,
                           max_length=128, return_tensors="pt", return_token_type_ids=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            ids = model(**inputs).logits.argmax(-1).cpu().tolist()
        out.extend(NLI_TO_SPEAKER[id2label[i]] for i in ids)
    return out


_SURFACE_DENY_WORDS = ("거짓", "틀린", "틀렸", "미신", "속설일", "사실이 아니", "사실 아니", "잘못", "근거 없", "근거가 없", "낭설",
                       "헛소문", "루머", "가짜", "뻥", "구라", "괴담", "도시전설", "오해", "아니래", "아니라")
_SURFACE_POINTERS = ("그 ", "그거", "그건", "이 ", "이거", "이건", "저 ", "걔", "거기", "해당")


def surface_rule(row: dict) -> str:
    """측정 기준선: 문장 뜻을 읽지 않고 겉모양만 보는 규칙."""
    sentence = row["sentence"]
    if sentence.startswith(_SURFACE_POINTERS):
        return "사실 없음"
    return "아니라고 함" if any(w in sentence for w in _SURFACE_DENY_WORDS) else "맞다고 함"


def report(name: str, pairs: list[tuple[dict, str]]) -> None:
    if not pairs:
        return
    correct = sum(1 for row, pred in pairs if pred == row["label"])
    flips = [(row, pred) for row, pred in pairs if {pred, row["label"]} == {"맞다고 함", "아니라고 함"}]
    print(f"{name}: 정확도 {correct}/{len(pairs)} ({correct / len(pairs):.0%}) | 맞다고↔아니라고 뒤바뀜 {len(flips)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--eval", required=True, type=Path)
    args = parser.parse_args()
    rows = json.loads(args.eval.read_text(encoding="utf-8"))
    preds = predict(args.model, rows)
    pairs = list(zip(rows, preds))

    print("===== 말한 사람 판별 평가 =====")
    report("전체", pairs)
    report("본 묶음", [p for p in pairs if p[0].get("subset", "본") == "본"])
    report("4B 정리 흠 묶음", [p for p in pairs if p[0].get("subset") == "4B정리흠"])
    for source in sorted({r.get("source", "") for r in rows}):
        report(f"출처 {source}", [p for p in pairs if p[0].get("source", "") == source])
    for label in SPEAKER_TO_NLI:
        report(f"정답 {label}", [p for p in pairs if p[0]["label"] == label])
    # 내용의 실제 참거짓(claim_true)이 적힌 평가 세트면, 참거짓 상식만으로는 못 맞히는 문항(거짓을 맞다고 함, 참을 아니라고 함)을
    # 따로 센다 - 이 점수가 올라야 문장을 제대로 읽게 된 것이다(2026-10-17: 참거짓만 알면 86%를 맞히는 평가 세트가 있었다).
    if any(r.get("claim_true") is not None for r in rows):
        hard = [p for p in pairs if (p[0]["label"], p[0].get("claim_true")) in {("맞다고 함", False), ("아니라고 함", True)}]
        easy = [p for p in pairs if (p[0]["label"], p[0].get("claim_true")) in {("맞다고 함", True), ("아니라고 함", False)}]
        report("상식으로는 못 맞히는 문항(거짓을 맞다고·참을 아니라고)", hard)
        report("상식과 같은 방향 문항", easy)
    # 문장을 읽지 않고 겉모양만 보는 규칙(평가 단어가 있으면 아니라고 함, 지시어로 시작하면 사실 없음, 나머지는 맞다고 함)이
    # 2026-10-17 평가 세트의 71.6%를 맞혔다. 이 규칙이 틀리는 문항에서의 점수가 문장을 제대로 읽는지 보여준다. 측정용 기준선일
    # 뿐 판정에는 쓰지 않는다.
    hard_surface = [p for p in pairs if surface_rule(p[0]) != p[0]["label"]]
    print(f"(겉모양 규칙 기준선 {len(pairs) - len(hard_surface)}/{len(pairs)})")
    report("겉모양 규칙으로는 못 맞히는 문항", hard_surface)
    # 상식(참이면 맞다고 함)으로도 겉모양 규칙으로도 못 맞히는 핵심 문항 - 문장 구조를 읽어야만 맞힌다.
    core = [p for p in hard_surface if (p[0]["label"], p[0].get("claim_true")) in {("맞다고 함", False), ("아니라고 함", True)}]
    report("상식·겉모양 둘 다로 못 맞히는 핵심 문항", core)
    # 같은 원문에 내용만 반대인 claim 쌍 - 둘 다 맞혀야 원문을 읽은 것이다.
    by_sentence: dict[str, list] = {}
    for row, pred in pairs:
        by_sentence.setdefault(row["sentence"], []).append((row, pred))
    groups = [g for g in by_sentence.values() if len(g) >= 2 and len({r["label"] for r, _ in g}) >= 2]
    both = sum(1 for g in groups if all(pred == r["label"] for r, pred in g))
    if groups:
        print(f"같은 원문의 라벨이 다른 claim 묶음 모두 맞힘: {both}/{len(groups)} ({both / len(groups):.0%})")
    for category in sorted({r.get("category", "") for r in rows if r.get("category")}):
        report(f"유형 {category}", [p for p in pairs if p[0].get("category") == category])
    confusion = collections.Counter((row["label"], pred) for row, pred in pairs)
    print("정답 -> 예측: " + ", ".join(f"{a}->{b} {n}" for (a, b), n in sorted(confusion.items())))
    print("\n틀린 항목(사람이 확인):")
    for row, pred in pairs:
        if pred != row["label"]:
            mark = "뒤바뀜" if {pred, row["label"]} == {"맞다고 함", "아니라고 함"} else "사실없음 혼동"
            print(f"  [{mark}] {row['sentence']} || {row['claim']} || 정답 {row['label']} / 예측 {pred} ({row.get('source', '')})")


if __name__ == "__main__":
    main()
