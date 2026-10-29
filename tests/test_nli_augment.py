from scripts.nli_augment import (
    CONTRADICTION,
    ENTAILMENT,
    EVAL_REFUTE,
    NEUTRAL,
    TRAIN_REFUTE,
    as_clause,
    build_examples,
)


def test_clause_from_verb_and_adjective_sentences():
    assert as_clause("선풍기를 켜고 자면 사망한다.") == "선풍기를 켜고 자면 사망한다는"
    assert as_clause("지구는 평평하다") == "지구는 평평하다는"


def test_clause_from_copula_follows_final_consonant():
    assert as_clause("부산은 대한민국의 수도이다.") == "부산은 대한민국의 수도라는"
    assert as_clause("고래는 어류의 한 종류이다.") == "고래는 어류의 한 종류라는"
    assert as_clause("세종대왕은 고려의 왕이었다.") == "세종대왕은 고려의 왕이었다는"
    assert as_clause("그 건물은 국보 제1호 문화재이다.") == "그 건물은 국보 제1호 문화재라는"
    assert as_clause("이 책의 저자는 김영식이다.") == "이 책의 저자는 김영식이라는"


def test_clause_skips_unusable_sentences():
    assert as_clause("10년 전에 나를 실망시켰던 영화.") is None  # '다'로 끝나지 않음
    assert as_clause("101빌딩 근처에서 즐길거리 찾기는 어렵습니다.") is None  # "어렵습니다는"은 비문
    assert as_clause("정말 그랬다고?") is None
    assert as_clause("짧다.") is None


def test_examples_teach_refutation_mention_and_affirmation():
    examples = build_examples(["선풍기를 켜고 자면 사망한다."] * 3, seed=0)

    labels = {e["label"] for e in examples if e["hypothesis"] == "선풍기를 켜고 자면 사망한다."}
    assert CONTRADICTION in labels and NEUTRAL in labels
    refuting = next(e for e in examples if e["label"] == CONTRADICTION)
    assert refuting["premise"].startswith("선풍기를 켜고 자면 사망한다는 ")
    assert len({e["premise"] for e in examples if e["label"] == CONTRADICTION}) == 1  # 같은 문장은 한 번만


def test_mention_or_refutation_as_claim_is_never_contradiction():
    """위험 오답 사례: "한국에는 … 속설이 있다"가 "… 근거 없는 미신이다"에 반박당하면 안 된다."""
    originals = [f"{i}번 다리는 강 위에 있다" for i in range(200)]
    examples = build_examples(originals, seed=3)

    # 가설이 원래 문장이 아니면, 소개·평가 문장 자체를 주장으로 둔 예시다.
    meta = [e for e in examples if e["hypothesis"] not in originals]
    assert meta
    assert all(e["label"] in (NEUTRAL, ENTAILMENT) for e in meta)


def test_held_out_examples_use_only_evaluation_templates():
    examples = build_examples(["지구는 평평하다."], seed=0, held_out=True)

    refuting = [e["premise"] for e in examples if e["label"] == CONTRADICTION]
    assert refuting and all(any(p == t.format(c="지구는 평평하다는") for t in EVAL_REFUTE) for p in refuting)
    assert not any(p == t.format(c="지구는 평평하다는") for t in TRAIN_REFUTE for p in refuting)
