from scripts.nli_check import MAX_SENTENCE_CHARS, NLI_THRESHOLD, candidate_sentences, confirm_sentences


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


def test_confirm_keeps_only_picked_sentences_reaching_threshold():
    picked = [("에펠탑", "에펠탑은 프랑스 파리에 있다."), ("대관람차", "대관람차는 박람회에서 처음 등장했다.")]
    score = _scorer({"에펠탑은 프랑스 파리에 있다.": 0.99, "대관람차는 박람회에서 처음 등장했다.": 0.3})

    assert confirm_sentences(score, picked, "에펠탑은 독일에 있다.") == [
        {"title": "에펠탑", "sentence": "에펠탑은 프랑스 파리에 있다.", "score": 0.99},
    ]


def test_confirm_threshold_is_inclusive_and_below_it_is_dropped():
    picked = [("A", "문장 하나입니다.")]

    assert confirm_sentences(_scorer({"문장 하나입니다.": NLI_THRESHOLD}), picked, "주장")
    assert confirm_sentences(_scorer({"문장 하나입니다.": NLI_THRESHOLD - 0.001}), picked, "주장") == []


def test_confirm_passes_core_claim_as_hypothesis():
    seen = []

    def score(premises, hypothesis):
        seen.append((premises, hypothesis))
        return [0.0] * len(premises)

    confirm_sentences(score, [("A", "충분히 긴 문장입니다.")], "핵심 주장")

    assert seen == [(["충분히 긴 문장입니다."], "핵심 주장")]


def test_confirm_with_nothing_picked_does_not_call_nli():
    def score(premises, hypothesis):
        raise AssertionError("호출되면 안 된다")

    assert confirm_sentences(score, [], "주장") == []
