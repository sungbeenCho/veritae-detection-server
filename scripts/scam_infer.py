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
    from paddleocr import PaddleOCR

    # PaddleOCR 3.x(PP-OCRv5)부터 .ocr(img, cls=True)가 deprecated -> .predict(img)로 교체.
    # predict()는 결과 객체 리스트를 반환하고, 인식된 텍스트는 rec_texts 키에 담겨 나온다
    # (2026-09-15, 3060Ti 실기에서 구버전 API 호출로 TypeError 발생해 확인 후 수정).
    # device="cpu": GPU(paddlepaddle-gpu, CUDA 13)로 시도했으나 faster-whisper가 쓰는
    # CUDA 12용 cuDNN과 같은 conda env 안에서 충돌(둘 다 nvidia-cudnn-cu1x가 필요해
    # 공존 불가). OCR은 이미지 한 장 처리라 CPU로도 충분히 빨라 CPU로 확정(2026-09-15).
    #
    # enable_mkldnn은 일부러 안 건드린다 - 처음엔 PIR/oneDNN 변환 크래시를 피하려고
    # enable_mkldnn=False를 넣었었는데, 그 크래시는 당시 cuDNN 파일이 cu12/cu13 충돌로
    # 꼬여있던 상태에서 난 것으로 보이고(재설치로 해결됨), enable_mkldnn=False로 둔
    # 상태에서 OCR 인식 결과가 전부 깨져서 나오는 문제가 실기에서 확인됨. PaddleOCR
    # 3.0.x~3.6.x 대에 enable_mkldnn 처리 자체에 알려진 회귀 버그가 있어
    # (PaddlePaddle/PaddleOCR#15632, #15782) 이 옵션에 아무 값이나 강제하는 것보다
    # 라이브러리 기본값에 맡기는 쪽이 안전하다고 판단(2026-09-15).
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, device="cpu")
    result = ocr.predict(str(image_path))
    if not result:
        return []
    return list(result[0]["rec_texts"])


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
