# scripts/nli_augment.py
"""NLI 보충 학습 데이터: 위키가 속설을 반박할 때 쓰는 평가형 문장을 KLUE 문장으로 자동 생성한다.

KLUE-NLI로만 학습한 모델은 "인간이 뇌의 10%만 사용한다는 것은 사실이 아니다"를 "인간은 뇌의 10%만 사용한다"와
모순이라고 보지 못했다(모순 확률 0.02, 2026-10-01 데스크탑 실측). "근거 없는 미신이다"(선풍기 사망설)도 마찬가지다.
KLUE에는 이런 평가 문장 형태가 거의 없어서 배운 적이 없는 것이다. 부정 표현에 약한 것은 NLI 모델에 흔한 약점이고,
해당 형태의 예시를 보충하는 것이 표준 처방이다.

예시는 KLUE 가설 문장 X로 만든다.
- "X라는 것은 사실이 아니다" 같은 평가 -> X와 모순
- "X라는 속설이 있다" 같은 소개 -> X와 중립(속설을 소개만 하는 참인 문장을 반박하지 않게 하는 반대 예시)
- "X라는 것은 사실이다" -> X와 함의("사실"이라는 단어만 보고 모순으로 찍는 지름길을 막는다)
- 소개·평가 문장 자체가 주장일 때 -> 모순이 아님(2026-10-01 위험 오답 사례: "한국에는 선풍기 … 속설이 있다")
평가용 틀(EVAL_*)은 학습용 틀과 겹치지 않게 두어, 틀을 외웠는지 새 표현에도 통하는지 가려낸다.
"""
from __future__ import annotations

import random

ENTAILMENT, NEUTRAL, CONTRADICTION = 0, 1, 2  # KLUE-NLI 라벨 순서(train_nli.LABELS)

TRAIN_REFUTE = [
    "{c} 것은 사실이 아니다.",
    "{c} 것은 근거 없는 미신이다.",
    "{c} 이야기는 잘못 알려진 것이다.",
    "{c} 주장은 과학적 근거가 없다.",
    "{c} 속설은 사실이 아닌 것으로 밝혀졌다.",
]
TRAIN_MENTION = [
    "{c} 속설이 있다.",
    "{c} 주장이 있다.",
    "{c} 이야기가 전해진다.",
    "{c} 소문이 돌았다.",
]
TRAIN_AFFIRM = [
    "{c} 것은 사실이다.",
    "{c} 것이 사실로 확인되었다.",
]

EVAL_REFUTE = ["{c} 것은 잘못된 정보다.", "{c} 것은 과학적으로 근거가 없는 속설이다."]
EVAL_MENTION = ["{c} 가설이 있다."]
EVAL_AFFIRM = ["{c} 것은 확인된 사실이다."]


def _has_final_consonant(char: str) -> bool:
    code = ord(char) - 0xAC00
    return 0 <= code <= 11171 and code % 28 != 0


def as_clause(sentence: str) -> str | None:
    """평서문을 "~다는/~라는" 관형절로 바꾼다("선풍기를 켜고 자면 사망한다." -> "… 사망한다는").
    '다'로 끝나지 않는 문장(의문문, 구어체 '~요' 등)이나 너무 짧거나 긴 문장은 쓰지 않는다(None)."""
    s = sentence.strip().rstrip(".").strip()
    if not (6 <= len(s) <= 60) or not s.endswith("다") or any(ch in s for ch in "?!\"'“”"):
        return None
    # "~습니다/~합니다"는 "어렵습니다는 속설이 있다"처럼 비문이 된다(2026-10-01 생성 예시 검토). 쓰지 않는다.
    if s.endswith("니다"):
        return None
    if s.endswith("이다") and len(s) > 3 and s[-3] != " ":
        stem = s[:-2]
        return stem + ("이라는" if _has_final_consonant(stem[-1]) else "라는")
    return s + "는"


def _examples(sentence: str, refute: list[str], mention: list[str], affirm: list[str], rng: random.Random) -> list[dict]:
    clause = as_clause(sentence)
    if clause is None:
        return []
    hypothesis = sentence.strip()
    refuting = rng.choice(refute).format(c=clause)
    mentioning = rng.choice(mention).format(c=clause)
    out = [
        {"premise": refuting, "hypothesis": hypothesis, "label": CONTRADICTION},
        {"premise": mentioning, "hypothesis": hypothesis, "label": NEUTRAL},
    ]
    if rng.random() < 0.5:
        out.append({"premise": rng.choice(affirm).format(c=clause), "hypothesis": hypothesis, "label": ENTAILMENT})
    if rng.random() < 0.5:
        # 소개·평가 문장 자체가 주장일 때: "X는 근거 없는 미신이다"는 "X라는 속설이 있다"나 그 자신과 모순이 아니다.
        out.append({"premise": refuting, "hypothesis": mentioning, "label": NEUTRAL})
        out.append({"premise": refuting, "hypothesis": rng.choice(refute).format(c=clause), "label": ENTAILMENT})
    return out


def build_examples(sentences: list[str], seed: int = 0, held_out: bool = False) -> list[dict]:
    """held_out=True면 학습에 쓰지 않은 평가용 틀로 만든다."""
    rng = random.Random(seed)
    refute, mention, affirm = (
        (EVAL_REFUTE, EVAL_MENTION, EVAL_AFFIRM) if held_out else (TRAIN_REFUTE, TRAIN_MENTION, TRAIN_AFFIRM)
    )
    seen: set[str] = set()
    examples: list[dict] = []
    for sentence in sentences:
        if sentence in seen:
            continue
        seen.add(sentence)
        examples.extend(_examples(sentence, refute, mention, affirm, rng))
    return examples
