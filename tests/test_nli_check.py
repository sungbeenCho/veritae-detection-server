import pytest

from scripts.nli_check import (
    MAX_SENTENCE_CHARS,
    NLI_THRESHOLD,
    candidate_sentences,
    label_indices,
    pick_best_contradiction,
)


def test_label_indices_reads_named_labels_in_any_order():
    assert label_indices({0: "CONTRADICTION", 1: "neutral", 2: "Entailment"}) == {
        "entailment": 2, "neutral": 1, "contradiction": 0,
    }


def test_label_indices_falls_back_to_klue_order_for_unnamed_labels():
    """라벨 이름이 LABEL_0.. 으로만 저장된 모델은 KLUE-NLI 순서(함의, 중립, 모순)를 따른다."""
    assert label_indices({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}) == {
        "entailment": 0, "neutral": 1, "contradiction": 2,
    }


def test_label_indices_rejects_a_model_without_all_three_labels():
    with pytest.raises(ValueError):
        label_indices({0: "contradiction", 1: "not_contradiction"})


def _scorer(table: dict[str, float]):
    return lambda premises, hypothesis: [table.get(p, 0.0) for p in premises]


def test_candidates_split_blocks_into_titled_sentences_in_order():
    blocks = [
        ("에펠탑", "에펠탑은 프랑스 파리에 있다. 1889년에 지어졌다."),
        ("대관람차", "대관람차는 박람회에서 처음 등장했다."),
    ]

    assert candidate_sentences(blocks) == [
        ("에펠탑", "에펠탑은 프랑스 파리에 있다."),
        ("에펠탑", "1889년에 지어졌다."),
        ("대관람차", "대관람차는 박람회에서 처음 등장했다."),
    ]


def test_candidates_skip_tiny_fragments_and_empty_blocks():
    assert candidate_sentences([("A", "가. 충분히 긴 문장입니다."), ("B", "")]) == [("A", "충분히 긴 문장입니다.")]


def test_candidates_skip_table_like_fragments_longer_than_a_real_sentence():
    """위키 표가 풀린 마침표 없는 긴 덩어리(2026-10-01: 에펠탑 복제품 목록)는 후보 문장으로 치지 않는다."""
    table = "에펠탑 미국 23 m 1:14 스케일 " * 40

    candidates = candidate_sentences([("에펠탑의 레플리카", table.strip() + ". 짧은 문장입니다.")])

    assert len(table) > MAX_SENTENCE_CHARS
    assert candidates == [("에펠탑의 레플리카", "짧은 문장입니다.")]


def test_pick_best_returns_the_highest_scoring_sentence_reaching_threshold():
    candidates = [("대관람차", "대관람차는 박람회에서 처음 등장했다."), ("에펠탑", "에펠탑은 프랑스 파리에 있다."), ("A", "무관한 문장입니다.")]
    score = _scorer({"에펠탑은 프랑스 파리에 있다.": 0.99, "대관람차는 박람회에서 처음 등장했다.": 0.9})

    best, scored = pick_best_contradiction(score, candidates, "에펠탑은 독일에 있다.")

    assert best == {"title": "에펠탑", "sentence": "에펠탑은 프랑스 파리에 있다.", "score": 0.99}
    assert [(s["title"], s["score"]) for s in scored] == [("대관람차", 0.9), ("에펠탑", 0.99), ("A", 0.0)]


def test_pick_best_threshold_is_inclusive_and_below_it_is_none():
    candidates = [("A", "문장 하나입니다.")]

    assert pick_best_contradiction(_scorer({"문장 하나입니다.": NLI_THRESHOLD}), candidates, "주장")[0] is not None
    assert pick_best_contradiction(_scorer({"문장 하나입니다.": NLI_THRESHOLD - 0.001}), candidates, "주장")[0] is None


def test_pick_best_returns_scores_even_when_nothing_reaches_threshold():
    """탐지가 비었을 때 NLI가 몇 점을 줬는지 알아야 원인을 안다(2026-10-13) - 탈락해도 점수는 돌려준다."""
    candidates = [("A 문서", "첫 문장입니다."), ("B 문서", "둘째 문장입니다.")]

    best, scored = pick_best_contradiction(lambda premises, hypothesis: [0.12, 0.40], candidates, "주장")

    assert best is None
    assert [(s["title"], s["sentence"], s["score"]) for s in scored] == [
        ("A 문서", "첫 문장입니다.", 0.12), ("B 문서", "둘째 문장입니다.", 0.40),
    ]


def test_pick_best_keeps_the_earlier_candidate_on_a_tie():
    """동점이면 앞 후보(주장과 의미가 더 가까운 문장)를 쓴다."""
    candidates = [("A", "먼저 나온 문장입니다."), ("B", "나중에 나온 문장입니다.")]

    best, _ = pick_best_contradiction(lambda premises, hypothesis: [0.95, 0.95], candidates, "주장")

    assert best["title"] == "A"


def test_pick_best_passes_core_claim_as_hypothesis_in_one_batch():
    seen = []

    def score(premises, hypothesis):
        seen.append((premises, hypothesis))
        return [0.0] * len(premises)

    pick_best_contradiction(score, [("A", "충분히 긴 문장입니다."), ("B", "또 다른 문장입니다.")], "핵심 주장")

    assert seen == [(["충분히 긴 문장입니다.", "또 다른 문장입니다."], "핵심 주장")]


def test_pick_best_with_no_candidates_does_not_call_nli():
    def score(premises, hypothesis):
        raise AssertionError("호출되면 안 된다")

    assert pick_best_contradiction(score, [], "주장") == (None, [])
