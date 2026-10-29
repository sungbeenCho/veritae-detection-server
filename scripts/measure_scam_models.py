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

기준점 다시 잡기(2026-10-27): 점수가 한쪽에 몰린 모델(kcelectra 0.97~0.98)은 0.5 기준이 무의미하다. 세트를 반으로
나눠 앞 절반에서 정상 오탐이 TARGET_FPR 이하가 되는 기준점을 고르고, 뒤 절반에서 합격 기준(설계서 6장: 정상 오탐
95%CI 상한 20% 이하, 사기 탐지 95%CI 하한 60% 이상, 각 40개 이상)으로 판정한다. 같은 데이터로 고르고 재면 부풀려진다.
- nli_zeroshot: 가짜정보용 NLI 모델(data/nli_model_aug)에 "~를 요구한다" 가설을 넣어 가장 높은 함의 확률을 점수로
  쓴다. 학습 데이터가 필요 없고 걸린 가설이 판정 이유가 된다.
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
    # user0074/voice-phishing-koelectra는 pytorch_model.bin만 있어 뺐다(pickle 로드는 torch 2.6 미만에서 거부, 2026-09-13에 이미 탈락).
    "xlmr_smishing": "donghyun95/kor-smishing-xlmroberta",
    "klue_smishing_synth": "donghyun95/smishing-detection-roberta-base",
    "kcelectra_smishing": "kdt-2-team4-newbiz/kcelectra-smishing-classifier",
    # 2026-10-27 추가 후보. 둘 다 모델 카드에 언어·학습 데이터·라이선스가 없다(같은 공개 데이터로 학습했을 가능성 높음).
    "nsjkm_electra": "nsjkm646588/smishing-classifier",
    "sseul2_distilbert": "sseul2/bert-smishing-model-final2",
    "nli_zeroshot": str(Path(__file__).resolve().parent.parent / "data" / "nli_model_aug"),
}
TARGET_FPR = 0.10  # 앞 절반에서 맞출 정상 오탐률. 뒤 절반 100개 기준 관측 10%면 95%CI 상한이 약 17%라 20% 기준에 여유가 남는다.
MIN_PER_CLASS = 40
# 사기 문자·통화가 공통으로 시키는 행동. 키워드가 아니라 NLI 모델이 뜻으로 판단한다.
NLI_HYPOTHESES = [
    "상대에게 돈을 보내거나 이체하라고 요구한다.",
    "상대에게 앱을 설치하거나 링크에 접속하라고 요구한다.",
    "상대에게 개인정보나 금융정보를 알려 달라고 요구한다.",
]
NLI_LABELS = ("entailment", "neutral", "contradiction")


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
        # text는 묶음 없이 통화 전체를 한 번에 읽는 후보(measure_scam_llm.py)용.
        return {"windows": windows, "text": " ".join(segments), "label": label}

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


def nli_entailment_index(model) -> int:
    # scripts/nli_check.label_indices와 같은 규칙. 그 모듈은 위키 색인 의존성(kiwipiepy 등)까지 불러와 여기선 쓰지 않는다.
    names = {i: str(n).lower() for i, n in model.config.id2label.items()}
    if all(n.startswith("label_") for n in names.values()):
        return NLI_LABELS.index("entailment")
    return next(int(i) for i, n in names.items() if n.startswith("entai"))


def score_nli(tokenizer, model, idx: int, texts: list[str], batch: int = 16) -> list[float]:
    best = np.zeros(len(texts))
    drop_token_type_ids = getattr(model.config, "type_vocab_size", 2) <= 1
    with torch.no_grad():
        for hyp in NLI_HYPOTHESES:
            out = []
            for i in range(0, len(texts), batch):
                chunk = texts[i:i + batch]
                enc = tokenizer(chunk, [hyp] * len(chunk), return_tensors="pt", truncation=True, max_length=256, padding=True)
                if drop_token_type_ids:
                    enc.pop("token_type_ids", None)
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                out += torch.softmax(model(**enc).logits, dim=-1)[:, idx].tolist()
            best = np.maximum(best, np.array(out))
    return best.tolist()


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z, p = 1.96, k / n
    mid, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(mid - half), float(mid + half)


def threshold_check(scores: np.ndarray, labels: np.ndarray) -> dict:
    """앞 절반에서 기준점을 고르고 뒤 절반에서 합격 기준으로 판정한다(라벨별로 섞은 뒤 반씩)."""
    rng = np.random.default_rng(SEED)
    pick, held = [], []
    for lab in (0, 1):
        idx = rng.permutation(np.where(labels == lab)[0])
        pick += idx[:len(idx) // 2].tolist()
        held += idx[len(idx) // 2:].tolist()
    normal = np.sort(scores[pick][labels[pick] == 0])[::-1]
    allowed = int(TARGET_FPR * len(normal))
    threshold = float(np.nextafter(normal[allowed], np.inf)) if allowed < len(normal) else float(normal[-1])
    hs, hl = scores[held], labels[held]
    scam, norm = hs[hl == 1], hs[hl == 0]
    tp, fp = int((scam >= threshold).sum()), int((norm >= threshold).sum())
    tpr_ci, fpr_ci = wilson(tp, len(scam)), wilson(fp, len(norm))
    if min(len(scam), len(norm)) < MIN_PER_CLASS:
        verdict = "판정 불가(개수 부족)"
    else:
        verdict = "합격" if fpr_ci[1] <= 0.20 and tpr_ci[0] >= 0.60 else "불합격"
    return {"threshold": threshold, "n_scam": len(scam), "n_normal": len(norm),
            "tpr": tp / len(scam), "tpr_ci": tpr_ci, "fpr": fp / len(norm), "fpr_ci": fpr_ci, "verdict": verdict}


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
        # HF 모델은 safetensors만 받는다. 로컬 NLI 폴더는 기본값(None, 있는 형식을 읽음)으로 둔다 -
        # False를 주면 Trainer가 저장한 model.safetensors를 건너뛰어 "파일 없음"으로 실패한다(2026-10-27).
        model = AutoModelForSequenceClassification.from_pretrained(model_id, use_safetensors=None if key == "nli_zeroshot" else True).eval().to(DEVICE)
        idx = nli_entailment_index(model) if key == "nli_zeroshot" else phishing_index(model)
        score = score_nli if key == "nli_zeroshot" else score_texts
        print(f"\n### {key} ({model_id}) id2label={model.config.id2label} -> 사기 칸 {idx}", flush=True)
        results[key] = {}
        for set_name, items in sets.items():
            if "windows" in items[0]:
                flat = [prepare(key, w) for it in items for w in it["windows"]]
                s = score(tokenizer, model, idx, flat)
                scores, pos = [], 0
                for it in items:
                    w = s[pos:pos + len(it["windows"])]
                    pos += len(it["windows"])
                    scores.append(max(w) * 0.7 + sum(w) / len(w) * 0.3)
            else:
                scores = score(tokenizer, model, idx, [prepare(key, it["text"]) for it in items])
            sc, lb = np.array(scores), np.array([it["label"] for it in items])
            lo, hi = auc_ci(sc, lb)
            tpr, fpr = float((sc[lb == 1] >= 0.5).mean()), float((sc[lb == 0] >= 0.5).mean())
            results[key][set_name] = {"auc": auc(sc, lb), "ci": [lo, hi], "tpr": tpr, "fpr": fpr,
                                      "scam_median": float(np.median(sc[lb == 1])), "normal_median": float(np.median(sc[lb == 0]))}
            r = results[key][set_name]
            print(f"  {set_name}: AUC {r['auc']:.2f} ({lo:.2f}~{hi:.2f}) | 0.5 기준 사기 탐지 {tpr:.0%} · 정상 오탐 {fpr:.0%} | 중앙값 사기 {r['scam_median']:.2f} / 정상 {r['normal_median']:.2f}", flush=True)
            t = r["threshold_check"] = threshold_check(sc, lb)
            print(f"    기준점 {t['threshold']:.4f}(앞 절반) -> 뒤 절반 사기 {t['n_scam']}·정상 {t['n_normal']}: 탐지 {t['tpr']:.0%}"
                  f" (하한 {t['tpr_ci'][0]:.0%}) · 오탐 {t['fpr']:.0%} (상한 {t['fpr_ci'][1]:.0%}) => {t['verdict']}", flush=True)
            r["scores"], r["labels"] = sc.tolist(), lb.tolist()
        print(f"  ({time.time() - t0:.0f}초)", flush=True)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    (DATA / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
