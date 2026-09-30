from scripts.nli_check import NLI_THRESHOLD, refuting_blocks


def _scorer(table: dict[str, float]):
    return lambda premises, hypothesis: [table.get(p, 0.0) for p in premises]


def test_keeps_only_blocks_whose_best_sentence_reaches_threshold():
    blocks = [
        ("에펠탑", "에펠탑은 프랑스 파리에 있다. 1889년에 지어졌다."),
        ("대관람차", "대관람차는 박람회에서 처음 등장했다."),
    ]
    score = _scorer({"에펠탑은 프랑스 파리에 있다.": 0.99, "대관람차는 박람회에서 처음 등장했다.": 0.3})

    confirmed = refuting_blocks(score, blocks, "에펠탑은 독일에 있다.")

    assert confirmed == [{
        "title": "에펠탑",
        "text": "에펠탑은 프랑스 파리에 있다. 1889년에 지어졌다.",
        "sentence": "에펠탑은 프랑스 파리에 있다.",
        "score": 0.99,
    }]


def test_threshold_is_inclusive_and_below_it_is_dropped():
    blocks = [("A", "문장 하나입니다.")]

    assert refuting_blocks(_scorer({"문장 하나입니다.": NLI_THRESHOLD}), blocks, "주장")
    assert refuting_blocks(_scorer({"문장 하나입니다.": NLI_THRESHOLD - 0.001}), blocks, "주장") == []


def test_passes_core_claim_as_hypothesis_and_skips_tiny_fragments():
    seen = []

    def score(premises, hypothesis):
        seen.append((premises, hypothesis))
        return [0.0] * len(premises)

    refuting_blocks(score, [("A", "가. 충분히 긴 문장입니다.")], "핵심 주장")

    assert seen == [(["충분히 긴 문장입니다."], "핵심 주장")]


def test_block_without_sentences_is_ignored():
    assert refuting_blocks(_scorer({}), [("A", "")], "주장") == []
