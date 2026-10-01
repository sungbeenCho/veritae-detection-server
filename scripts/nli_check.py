"""반박 확인 단계: 근거 문장 선택(LLM)이 고른 문장이 정말 주장과 모순되는지, 문장 두 개의 모순 여부만
판정하도록 학습된 NLI 분류기로 확인한다(전체 흐름은 misinfo_lib.py 모듈 설명 참고).

LLM에게 다시 묻는 방식은 같은 모델이 같은 실수를 반복하고 문구를 바꿀 때마다 오류가 옮겨 다녀 쓸 수
없었다(2026-10-01 데스크탑 실측 3회). NLI 분류기는 판단 방식이 달라 LLM과 독립적이고, 같은 입력이면 항상
같은 결과를 낸다. 대신 NLI는 주어만 같고 내용이 무관한 문장에도 높은 모순 점수를 줘서("지구는 평평하다" vs
"지구의는 받침대에 베어링이 있다" 0.989) 근거 문장을 고르는 일은 맡기지 않고, LLM이 고른 문장의 확인만 맡긴다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from wiki_index import split_sentences

# 이 값 이상이면 모순으로 본다. 2026-10-01 노트북 시험(klue/roberta-large KLUE-NLI 모델, 실제 위키 문단)에서
# 진짜 반박 문장은 0.81~0.999, 반박이 아닌 문장은 대부분 0.05 미만이었다.
NLI_THRESHOLD = 0.8

# 이보다 긴 "문장"은 확인하지 않는다. 위키 표가 글자로 풀리면 마침표 없이 수백~천 자 덩어리가 되는데
# (예: 에펠탑 복제품 목록 표), NLI가 여기에 반응해 표가 반박 근거로 표시됐다(2026-10-01 데스크탑 실측).
# 실제 근거 문장은 길어야 200~300자 수준이다.
MAX_SENTENCE_CHARS = 300

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


def candidate_sentences(blocks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """인용 문단을 (문서 제목, 문장) 목록으로 나눈다. 근거 문장 선택(LLM)에 번호를 매겨 보여줄 후보다.
    너무 짧은 조각과 표가 풀린 긴 덩어리(MAX_SENTENCE_CHARS 초과)는 문장으로 치지 않는다."""
    return [
        (title, sentence)
        for title, text in blocks
        for sentence in split_sentences(text)
        if 4 < len(sentence) <= MAX_SENTENCE_CHARS
    ]


def confirm_sentences(
    score: ContradictionScorer, picked: list[tuple[str, str]], claim: str, threshold: float = NLI_THRESHOLD,
) -> list[dict]:
    """근거 문장 선택(LLM)이 고른 문장만 NLI로 확인해, 모순 확률이 threshold 이상인 것만 돌려준다.
    결과: [{"title", "sentence", "score"}] - 입력 순서 유지."""
    if not picked:
        return []
    scores = score([sentence for _, sentence in picked], claim)
    return [
        {"title": title, "sentence": sentence, "score": value}
        for (title, sentence), value in zip(picked, scores)
        if value >= threshold
    ]


def default_model_path() -> str:
    """detection-server 레포의 data/nli_model(scripts/train_nli.py 출력 위치)."""
    return str(Path(__file__).resolve().parent.parent / "data" / "nli_model")
