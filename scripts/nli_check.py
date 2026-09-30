"""반박 확인 단계: 1차 판정(LLM)이 반박으로 인용한 근거 문단을 문장 단위로 나눠, 문장 두 개의 모순 여부만
판정하도록 학습된 NLI 분류기로 "이 문장이 주장과 모순되는가"를 확인한다.

LLM에게 다시 묻는 방식은 같은 모델이 같은 실수를 반복하고 문구를 바꿀 때마다 오류가 옮겨 다녀 쓸 수
없었다(2026-10-01 데스크탑 실측 3회). NLI 분류기는 판단 방식이 달라 LLM과 독립적이고, 같은 입력이면 항상
같은 결과를 낸다. 대신 NLI도 혼자서는 참인 문장에 오판할 수 있어("세종대왕은 조선의 왕이다" vs 같은 문단의
"중국 세종은 상나라 왕의 묘호") LLM이 반박이라고 한 문장에만 쓰고, 둘 다 반박이라고 할 때만 표시한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from wiki_index import split_sentences

# 이 값 이상이면 모순으로 본다. 2026-10-01 노트북 시험(klue/roberta-large KLUE-NLI 모델, 실제 위키 문단)에서
# 진짜 반박 문장은 0.81~0.999, 반박이 아닌 문장은 대부분 0.05 미만이었다.
NLI_THRESHOLD = 0.8

ContradictionScorer = Callable[[list[str], str], list[float]]


class NliModel:
    """모델은 처음 쓸 때 한 번만 올린다 - 반박 후보가 없는 요청은 로딩 비용을 내지 않는다."""

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_path).eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            labels = {i: str(label).lower() for i, label in model.config.id2label.items()}
            if not any("contra" in label for label in labels.values()):
                # 라벨 이름이 LABEL_0.. 으로만 저장된 모델은 KLUE-NLI 순서(함의, 중립, 모순)를 따른다.
                labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
            contradiction = next(i for i, label in labels.items() if "contra" in label)
            drop_token_type_ids = getattr(model.config, "type_vocab_size", 2) <= 1
            self._loaded = (torch, tokenizer, model, device, contradiction, drop_token_type_ids)
        return self._loaded

    def contradiction_scores(self, premises: list[str], hypothesis: str) -> list[float]:
        if not premises:
            return []
        torch, tokenizer, model, device, contradiction, drop_token_type_ids = self._load()
        inputs = tokenizer(
            premises, [hypothesis] * len(premises),
            padding=True, truncation=True, max_length=256, return_tensors="pt",
        )
        if drop_token_type_ids:
            inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
        return probs[:, contradiction].cpu().tolist()


def refuting_blocks(
    score: ContradictionScorer, blocks: list[tuple[str, str]], claim: str, threshold: float = NLI_THRESHOLD,
) -> list[dict]:
    """각 문단에서 주장과 가장 모순되는 문장을 찾아, 모순 확률이 threshold 이상인 문단만 돌려준다.
    결과: [{"title", "text", "sentence", "score"}] - 입력 순서 유지."""
    confirmed = []
    for title, text in blocks:
        sentences = [s for s in split_sentences(text) if len(s) > 4]
        scores = score(sentences, claim)
        if not scores:
            continue
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] >= threshold:
            confirmed.append({"title": title, "text": text, "sentence": sentences[best], "score": scores[best]})
    return confirmed


def default_model_path() -> str:
    """detection-server 레포의 data/nli_model(scripts/train_nli.py 출력 위치)."""
    return str(Path(__file__).resolve().parent.parent / "data" / "nli_model")
