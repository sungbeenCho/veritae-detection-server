"""사기감지 후보 모델 비교 시험(2026-10-18) - 데스크탑 GPU(nli-train 환경)용.

Lilju 대체 후보(HuggingFace 한국어 스미싱 모델)를 같은 시험 세트로 비교한다. 서버 코드는 바꾸지 않는다.
공개 데이터는 재배포 금지 조건이 있어 저장소에 넣지 않고, 처음 실행 때 data/scam_eval/(git 제외)에 직접 내려받는다.
사용: conda activate nli-train; cd scripts; python measure_scam_models.py

세트
- sms_real: 실제 정상 문자(meal-bbang class 3, 택배·쇼핑 알림 등) vs 실제 스미싱(Ez-Sy01 class 1). 각 200개 무작위.
  주의: donghyun95/kor-smishing-xlmroberta는 이 두 데이터로 학습했다고 모델 카드에 적혀 있어 점수가 부풀려진다.
- hard_msg: 헷갈리는 문자(작성자 G·H, 정상/사기).
- hard_call: 헷갈리는 통화체(작성자 I).
- call_real: KorCCVi 시험 분할 실제 통화 142통(금감원 사기 71 / AI Hub 정상 71). 정상은 금융과 무관한 대화만 골라졌다(저자 문서).
통화는 서버 설계대로 연속 3문장(전사 조각)을 한 칸씩 밀어 묶고, 묶음 점수에 지금 서버 집계식(최댓값x0.7+평균x0.3)을 쓴다.

출력: 세트별 AUC(95% 부트스트랩 구간), 0.5 기준 사기 탐지율·정상 오탐률, 점수 분포 요약.
"""
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent.parent / "data" / "scam_eval"
DATA = HERE
HARD = Path(__file__).resolve().parent.parent / "tests" / "scam"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOWNLOADS = {
    "korccvi_test_segment_manifest.jsonl": "https://raw.githubusercontent.com/selfcontrol7/Korean_Voice_Phishing_Detection/main/Multimodal/data/test_segment_manifest.jsonl",
    "meal_bbang.csv": "https://huggingface.co/datasets/meal-bbang/Korean_message/resolve/main/lgaidataset.csv",
    "ezsy_raw.zip": "https://raw.githubusercontent.com/Ez-Sy01/KOR_phishing_Detect-Dataset/main/data/KOR_phishing_data_raw.zip",
}


def download() -> None:
    import urllib.request
    import zipfile
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS.items():
        if not (DATA / name).exists():
            print(f"내려받는 중: {url}", flush=True)
            urllib.request.urlretrieve(url, DATA / name)
    if not (DATA / "ezsy" / "KOR_phishing_data_raw.csv").exists():
        zipfile.ZipFile(DATA / "ezsy_raw.zip").extractall(DATA / "ezsy")
SEED = 0
SMS_PER_CLASS = 200
WINDOW = 3
CALLS_PER_CLASS = 71  # KorCCVi 시험 분할 전체(라벨별 71통)
BOOTSTRAP = 1000
# 라벨 검토 2명이 "글만으로는 판단 불가"로 본 항목(2026-10-18) - 시험에서 뺀다.
AMBIGUOUS = {"G04", "G09", "G23", "G29", "G39", "H02", "H29", "I30"}
URL = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(com|net|kr|co\.kr|ly|gl|me|io|example)\S*)")

MODELS = {
    "lilju": "Lilju/voicephishing_kobert",
    "user0074": "user0074/voice-phishing-koelectra",
    "xlmr_smishing": "donghyun95/kor-smishing-xlmroberta",
    "klue_smishing_synth": "donghyun95/smishing-detection-roberta-base",
    "kcelectra_smishing": "kdt-2-team4-newbiz/kcelectra-smishing-classifier",
}


def prepare(model_key: str, text: str) -> str:
    # 모델 카드가 권한 입력 형태: kcelectra는 URL을 <URL> 토큰으로 바꿔 학습했다.
    if model_key == "kcelectra_smishing":
        return URL.sub("<URL>", text)
    return text


def load_sets() -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    meal = [r for r in csv.DictReader(open(DATA / "meal_bbang.csv", encoding="utf-8-sig")) if r["class"] == "3"]
    ez = [r for r in csv.DictReader(open(DATA / "ezsy" / "KOR_phishing_data_raw.csv", encoding="utf-8-sig")) if r["class"] == "1"]
    sms = [{"text": r["content"], "label": 0} for r in rng.sample(meal, SMS_PER_CLASS)]
    sms += [{"text": r["content"], "label": 1} for r in rng.sample(ez, SMS_PER_CLASS)]
    sets = {"sms_real": sms}

    hard_msg = []
    for name in ("eval_hard_G.json", "eval_hard_H.json"):
        path = HARD / name
        if path.exists():
            hard_msg += [{"text": r["text"], "label": 1 if r["label"] == "scam" else 0} for r in json.loads(path.read_text(encoding="utf-8")) if r["id"] not in AMBIGUOUS]
    if hard_msg:
        sets["hard_msg"] = hard_msg

    def call_item(segments: list[str], label: int) -> dict:
        # 서버 설계(설계서 3.1)와 같이 한 칸씩 밀며 3문장씩 묶는다.
        windows = [" ".join(segments[i:i + WINDOW]) for i in range(max(1, len(segments) - WINDOW + 1))]
        return {"windows": windows, "label": label}

    path = HARD / "eval_hard_I.json"
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        sets["hard_call"] = [call_item([s for s in re.split(r"(?<=[.?!])\s+", r["text"]) if s.strip()], 1 if r["label"] == "scam" else 0) for r in rows if r["id"] not in AMBIGUOUS]

    calls = defaultdict(list)
    labels = {}
    for line in open(DATA / "korccvi_test_segment_manifest.jsonl", encoding="utf-8"):
        s = json.loads(line)
        calls[s["call_id"]].append((s["start"], s["text"]))
        labels[s["call_id"]] = s["label"]
    ids = sorted(calls)
    picked = [k for lab in (0, 1) for k in rng.sample([k for k in ids if labels[k] == lab], CALLS_PER_CLASS)]
    sets["call_real"] = [call_item([t for _, t in sorted(calls[k])], labels[k]) for k in picked]
    return sets


def phishing_index(model) -> int:
    names = {i: str(n).lower() for i, n in model.config.id2label.items()}
    for i, n in names.items():
        if any(w in n for w in ("phish", "spam", "smish", "scam", "fraud", "1")) and n not in ("label_0", "0"):
            return int(i)
    return 1


def score_texts(tokenizer, model, idx: int, texts: list[str], batch: int = 16) -> list[float]:
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tokenizer(texts[i:i + batch], return_tensors="pt", truncation=True, max_length=256, padding=True).to(DEVICE)
            enc.pop("token_type_ids", None) if model.config.model_type in ("roberta", "xlm-roberta") else None
            probs = torch.softmax(model(**enc).logits, dim=-1)
            out += probs[:, idx].tolist()
    return out


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def auc_ci(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, len(labels), len(labels))
        if len(set(labels[i])) == 2:
            vals.append(auc(scores[i], labels[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    download()
    print(f"장치: {DEVICE}", flush=True)
    sets = load_sets()
    print({k: (len(v), sum(x["label"] for x in v)) for k, v in sets.items()}, "(개수, 사기 수)", flush=True)
    results = {}
    for key, model_id in MODELS.items():
        if len(sys.argv) > 1 and key not in sys.argv[1:]:
            continue
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, use_safetensors=None).eval().to(DEVICE)
        idx = phishing_index(model)
        print(f"\n### {key} ({model_id}) id2label={model.config.id2label} -> 사기 칸 {idx}", flush=True)
        results[key] = {}
        for set_name, items in sets.items():
            if "windows" in items[0]:
                flat = [prepare(key, w) for it in items for w in it["windows"]]
                s = score_texts(tokenizer, model, idx, flat)
                scores, pos = [], 0
                for it in items:
                    w = s[pos:pos + len(it["windows"])]
                    pos += len(it["windows"])
                    scores.append(max(w) * 0.7 + sum(w) / len(w) * 0.3)
            else:
                scores = score_texts(tokenizer, model, idx, [prepare(key, it["text"]) for it in items])
            sc, lb = np.array(scores), np.array([it["label"] for it in items])
            lo, hi = auc_ci(sc, lb)
            tpr, fpr = float((sc[lb == 1] >= 0.5).mean()), float((sc[lb == 0] >= 0.5).mean())
            results[key][set_name] = {"auc": auc(sc, lb), "ci": [lo, hi], "tpr": tpr, "fpr": fpr,
                                      "scam_median": float(np.median(sc[lb == 1])), "normal_median": float(np.median(sc[lb == 0]))}
            r = results[key][set_name]
            print(f"  {set_name}: AUC {r['auc']:.2f} ({lo:.2f}~{hi:.2f}) | 0.5 기준 사기 탐지 {tpr:.0%} · 정상 오탐 {fpr:.0%} | 중앙값 사기 {r['scam_median']:.2f} / 정상 {r['normal_median']:.2f}", flush=True)
        print(f"  ({time.time() - t0:.0f}초)", flush=True)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    (DATA / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
