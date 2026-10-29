"""반박 확인·근거 선택 단계: LLM이 1차로 "반박"이라고 판정한 문장에 대해, 위키 문단의 문장 중 주장과 모순인 것을
문장 두 개의 모순 여부만 판정하도록 학습된 NLI 분류기로 찾는다(전체 흐름은 misinfo_lib.py 모듈 설명 참고).

LLM에게 다시 묻는 방식은 같은 모델이 같은 실수를 반복하고 문구를 바꿀 때마다 오류가 옮겨 다녀 쓸 수
없었다(2026-10-01 데스크탑 실측 3회). NLI 분류기는 판단 방식이 달라 LLM과 독립적이고, 같은 입력이면 항상
같은 결과를 낸다.

근거 문장도 NLI 점수로 고른다(2026-10-13). 10-01에는 NLI가 주어만 같고 내용이 무관한 문장에도 높은 점수를 줘서
("지구는 평평하다" vs "지구의는 받침대에 베어링이 있다" 0.989) LLM이 근거 문장을 고르게 하고 NLI는 확인만 시켰는데,
LLM이 엉뚱한 문장을 골라 맞는 반박 7개를 버렸다(59문장 기록 대조). 대신 주장과 의미가 가까운 상위 15개 문장만
NLI로 채점하고, LLM의 "반박" 판정이 먼저 있어야 한다는 조건으로 무관한 문장이 표시되는 위험을 줄였다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from misinfo_timing import logged_step
from wiki_index import split_sentences

# 이 값 이상이면 모순으로 본다. 2026-10-01 노트북 시험(klue/roberta-large KLUE-NLI 모델, 실제 위키 문단)에서
# 진짜 반박 문장은 0.81~0.999, 반박이 아닌 문장은 대부분 0.05 미만이었다.
NLI_THRESHOLD = 0.8

# 이보다 긴 "문장"은 확인하지 않는다. 위키 표가 글자로 풀리면 마침표 없이 수백~천 자 덩어리가 되는데
# (예: 에펠탑 복제품 목록 표), NLI가 여기에 반응해 표가 반박 근거로 표시됐다(2026-10-01 데스크탑 실측).
# 실제 근거 문장은 길어야 200~300자 수준이다.
MAX_SENTENCE_CHARS = 300

ContradictionScorer = Callable[[list[str], str], list[float]]

LABELS = ("entailment", "neutral", "contradiction")


def label_indices(id2label: dict[int, str]) -> dict[str, int]:
    """모델 설정의 라벨 이름으로 함의/중립/모순의 출력 위치를 찾는다. 라벨 이름이 LABEL_0.. 으로만 저장된
    모델은 KLUE-NLI 순서(함의, 중립, 모순)를 따른다."""
    names = {i: str(label).lower() for i, label in id2label.items()}
    if all(name.startswith("label_") for name in names.values()) and len(names) == len(LABELS):
        return {label: i for i, label in enumerate(LABELS)}
    found = {label: next((i for i, name in names.items() if name.startswith(label[:5])), None) for label in LABELS}
    if None in found.values():
        raise ValueError(f"함의/중립/모순 세 라벨을 가진 NLI 모델이 아니다: {id2label}")
    return found


class NliModel:
    """모델은 처음 쓸 때 한 번만 올린다 - 반박 후보가 없는 요청은 로딩 비용을 내지 않는다."""

    def __init__(self, model_path: str, load_step: str = "load_nli"):
        self._model_path = model_path
        self._load_step = load_step
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            # 반박이 나온 문장에서 처음 쓸 때 읽는다 - 이 시간을 서버 로그에서 볼 수 있게 잰다(2026-10-13).
            with logged_step(self._load_step):
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self._model_path)
                model = AutoModelForSequenceClassification.from_pretrained(self._model_path).eval()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                indices = label_indices(model.config.id2label)
                drop_token_type_ids = getattr(model.config, "type_vocab_size", 2) <= 1
                self._loaded = (torch, tokenizer, model, device, indices, drop_token_type_ids)
        return self._loaded

    def label_probabilities(self, premises: list[str], hypothesis: str) -> list[dict[str, float]]:
        """전제마다 {"entailment", "neutral", "contradiction"} 확률."""
        if not premises:
            return []
        torch, tokenizer, model, device, indices, drop_token_type_ids = self._load()
        inputs = tokenizer(
            premises, [hypothesis] * len(premises),
            padding=True, truncation=True, max_length=256, return_tensors="pt",
        )
        if drop_token_type_ids:
            inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().tolist()
        return [{label: row[i] for label, i in indices.items()} for row in probs]

    def contradiction_scores(self, premises: list[str], hypothesis: str) -> list[float]:
        return [p["contradiction"] for p in self.label_probabilities(premises, hypothesis)]

    def entailment_scores(self, premises: list[str], hypothesis: str) -> list[float]:
        """아니라고 한 내용이 참인 문장("지구가 둥글다는 건 거짓말이래요")의 근거는 그 내용을 뒷받침하는 문장이다."""
        return [p["entailment"] for p in self.label_probabilities(premises, hypothesis)]


def candidate_sentences(blocks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """검색된 문단을 (문서 제목, 문장) 목록으로 나눈다. 근거 문장 선택(e5로 추린 뒤 LLM)의 후보다.
    너무 짧은 조각과 표가 풀린 긴 덩어리(MAX_SENTENCE_CHARS 초과)는 문장으로 치지 않는다."""
    return [
        (title, sentence)
        for title, text in blocks
        for sentence in split_sentences(text)
        if 4 < len(sentence) <= MAX_SENTENCE_CHARS
    ]


def pick_best_contradiction(
    score: ContradictionScorer, candidates: list[tuple[str, str]], claim: str, threshold: float = NLI_THRESHOLD,
) -> tuple[dict | None, list[dict]]:
    """후보 문장 전부에 NLI 점수를 매겨 (모순 점수가 가장 높고 threshold 이상인 문장 하나 또는 None, 점수가 매겨진 전체)를
    돌려준다. 점수가 같으면 앞 후보(주장과 의미가 더 가까운 문장)를 쓴다.
    탈락했을 때도 점수를 돌려주는 이유: 탐지가 비었을 때 NLI가 몇 점을 줬는지 알아야 원인을 찾는다(2026-10-13)."""
    if not candidates:
        return None, []
    scores = score([sentence for _, sentence in candidates], claim)
    scored = [
        {"title": title, "sentence": sentence, "score": value}
        for (title, sentence), value in zip(candidates, scores)
    ]
    best = max(scored, key=lambda item: item["score"])
    return (best if best["score"] >= threshold else None), scored


def default_model_path() -> str:
    """detection-server 레포의 data/nli_model_aug(scripts/train_nli.py --init-model --augment 출력 위치).
    app/config.py의 MISINFO_NLI_MODEL 기본값과 같아야 한다 - 회귀 테스트와 실서버가 같은 모델로 돌게
    tests/test_config.py가 둘을 비교한다."""
    return str(Path(__file__).resolve().parent.parent / "data" / "nli_model_aug")
