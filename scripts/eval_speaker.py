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
    confusion = collections.Counter((row["label"], pred) for row, pred in pairs)
    print("정답 -> 예측: " + ", ".join(f"{a}->{b} {n}" for (a, b), n in sorted(confusion.items())))
    print("\n틀린 항목(사람이 확인):")
    for row, pred in pairs:
        if pred != row["label"]:
            mark = "뒤바뀜" if {pred, row["label"]} == {"맞다고 함", "아니라고 함"} else "사실없음 혼동"
            print(f"  [{mark}] {row['sentence']} || {row['claim']} || 정답 {row['label']} / 예측 {pred} ({row.get('source', '')})")


if __name__ == "__main__":
    main()
