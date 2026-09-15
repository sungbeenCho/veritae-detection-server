#!/usr/bin/env python
"""scam_infer.py

이미지(OCR) 또는 음성(STT)에서 텍스트를 추출해 Lilju/voicephishing_kobert로 문장별
피싱 확률을 매기고, 하나의 위험도 점수 + 근거 문장 목록을 JSON으로 저장한다.

`text-extraction` conda env(PaddleOCR, faster-whisper, transformers, kss 설치됨)에서
실행되어야 한다. veritae-detection-server(FastAPI, detection-api env)는 이 스크립트를
subprocess로 호출하고 --output 경로의 JSON만 읽는다 - 무거운 의존성을 FastAPI
프로세스에 넣지 않기 위함(SPAI/AntiDeepfake/dfdc 연동과 동일한 패턴).

docs(veritae-server 레포): docs/superpowers/specs/2026-09-13-fraud-risk-analysis-design.md
"""
import argparse
import json
from pathlib import Path

EVIDENCE_SCORE_THRESHOLD = 0.5  # spai_runner.py/antideepfake_infer.py/dfdc_infer.py와 동일 임계값
MAX_SENTENCE_LENGTH = 300  # Lilju 학습 시 입력 길이를 크게 벗어나는 극단값 방어용 - 실측 근거 없는 안전장치
PHISHING_LABEL_INDEX = 1  # Lilju 모델카드 없음 - 2026-09-13 실측(9/10 정확도)으로 확인된 값


def extract_text_units_ocr(image_path: Path, lang: str) -> list[str]:
    import easyocr

    # PaddleOCR(PP-OCRv3/v5 둘 다)에서 3060Ti 실기 검증 중 심각한 문제들이 발견돼
    # EasyOCR로 교체했다(2026-09-15):
    # - PP-OCRv5(3.x): CPU 실행 시 PIR/oneDNN 크래시, GPU는 faster-whisper용 cuDNN과
    #   버전 충돌(같은 conda env 안에 CUDA 12/13용 cuDNN이 공존 불가).
    # - PP-OCRv3(2.x, 안정성 위해 다운그레이드): 크래시는 없지만 "국민은행" 같은 특정
    #   단어를 일관되게 다른 글자로 잘못 인식(예: "우긍금논") - 같은 이미지로 EasyOCR과
    #   직접 비교해 실측 확인, EasyOCR 쪽이 명확히 더 정확했음.
    # gpu=False: 이 프로세스에서 GPU를 쓰면 faster-whisper의 cuDNN과 다시 충돌할 위험이
    # 있어, 이미지 한 장 처리라 속도 손해가 적은 CPU로 고정한다(PaddleOCR 때와 동일 판단).
    reader = easyocr.Reader([lang, "en"], gpu=False)
    return reader.readtext(str(image_path), detail=0)


def extract_text_units_stt(audio_path: Path, model_size: str) -> list[str]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size)
    segments, _info = model.transcribe(str(audio_path), language="ko")
    return [segment.text.strip() for segment in segments if segment.text.strip()]


def split_sentences(text_units: list[str]) -> list[str]:
    """native 단위(OCR 줄/STT segment) 각각을 kss로 한 번씩 통과시킨다. 쪼갤 게 없으면
    원래 그대로 1개가 나오므로 "길면 쪼갠다"는 조건 분기가 필요 없다. kss는 문어체를
    가정하고 만들어져 STT/OCR의 노이즈 낀 실제 출력에서 어떻게 반응할지 데스크탑
    실측 전이므로, 실패 시 원문 그대로 폴백한다(2026-09-13 설계)."""
    import kss

    sentences: list[str] = []
    for unit in text_units:
        if not unit:
            continue
        try:
            sentences.extend(s for s in kss.split_sentences(unit) if s.strip())
        except Exception:
            sentences.append(unit)
    return [s[:MAX_SENTENCE_LENGTH] for s in sentences]


def score_sentences(sentences: list[str], model_id: str) -> list[float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, use_safetensors=True)
    model.eval()

    scores = []
    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            scores.append(probs[0, PHISHING_LABEL_INDEX].item())
    return scores


def aggregate(scores: list[float]) -> float:
    """사용자 확정(2026-09-13): 최고확률x0.7 + 평균확률x0.3 (Lilju 자체 데모 app.py의
    집계 방식을 그대로 채택 - 모델의 일부가 아닌 후처리 로직이라 자유롭게 채택 가능함을
    확인하고 결정함, spec §4)."""
    return max(scores) * 0.7 + (sum(scores) / len(scores)) * 0.3


def build_evidence(sentences: list[str], scores: list[float]) -> list[dict]:
    return [
        {"sentence": sentence, "score": score}
        for sentence, score in zip(sentences, scores)
        if score >= EVIDENCE_SCORE_THRESHOLD
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["ocr", "stt"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lilju-model-id", required=True)
    parser.add_argument("--paddleocr-lang", required=True)
    parser.add_argument("--whisper-model-size", required=True)
    args = parser.parse_args()

    if args.mode == "ocr":
        text_units = extract_text_units_ocr(args.input, args.paddleocr_lang)
        # OCR 엔진(PaddleOCR/EasyOCR 공통)이 한 줄 단위를 넘어 단어/구 단위로도 잘게
        # 쪼개 인식하는 경우가 많아(2026-09-15 3060Ti 실기 확인 - 2026-09-13 설계 당시
        # 가정과 달랐음), 감지된 조각을 kss에 각각 따로 넣으면 합칠 문장 자체가 없어
        # 단어 단위로 그대로 나온다. 조각을 전부 하나로 합친 뒤 kss에 넣어야 문장부호
        # 기준으로 실제 문장 단위가 복원된다.
        # STT(음성) segment는 이미 발화 간 쉬는 구간 기준이라 문장에 가까워 그대로 둔다.
        text_units = [" ".join(text_units)] if text_units else []
    else:
        text_units = extract_text_units_stt(args.input, args.whisper_model_size)

    sentences = split_sentences(text_units)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not sentences:
        args.output.write_text(json.dumps({"score": None, "evidence": []}), encoding="utf-8")
        return

    scores = score_sentences(sentences, args.lilju_model_id)
    result = {"score": aggregate(scores), "evidence": build_evidence(sentences, scores)}
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
