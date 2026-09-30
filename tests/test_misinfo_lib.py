import json

from scripts.misinfo_lib import (
    JUDGE_SCHEMA,
    PREMISE_MAX_CHARS,
    VERIFY_SCHEMA,
    build_claim,
    build_judge_prompt,
    build_verify_prompt,
    fit_blocks,
    parse_judge_response,
    parse_verify_response,
    replace_block_numbers,
)


def test_fit_blocks_keeps_everything_under_the_limit():
    blocks = [("A", "가" * 600), ("B", "나" * 600)]

    assert fit_blocks(blocks, max_chars=5000) == blocks


def test_fit_blocks_drops_whole_lower_ranked_blocks_without_cutting_text():
    blocks = [("A", "가" * 3000), ("B", "나" * 2500), ("C", "다" * 1000)]

    assert fit_blocks(blocks, max_chars=5000) == [("A", "가" * 3000), ("C", "다" * 1000)]


def test_fit_blocks_cuts_only_when_the_top_block_alone_is_too_long():
    kept = fit_blocks([("A", "가" * 9000), ("B", "나" * 10)], max_chars=5000)

    assert kept == [("A", "가" * 4999)]


def test_worst_case_prompt_fits_context_even_at_one_token_per_character():
    """입력이 컨텍스트를 넘으면 오반박을 막는 지시문부터 잘린다 - 글자 수가 곧 토큰 수라고 가정해도
    (한국어 토큰은 한 글자보다 짧지 않다) 판정 출력 여유 1,000토큰을 남기고 8192 안에 들어가야 한다."""
    worst = fit_blocks([("제목" * 10, "가" * 3000)] * 5)
    sentence = "가" * 300

    for prompt in (build_judge_prompt(sentence, worst), build_verify_prompt(sentence, worst)):
        assert len(prompt) + 1000 <= 8192
    assert sum(len(t) + len(x) for t, x in worst) <= PREMISE_MAX_CHARS


def test_replace_block_numbers_uses_titles_and_leaves_unknown_numbers():
    blocks = [("백신", "..."), ("자폐증", "...")]

    assert replace_block_numbers("[1] 문서와 [2] 문서, [7]은 모름", blocks) == "'백신' 문서와 '자폐증' 문서, [7]은 모름"


def _judge_raw(**overrides) -> str:
    data = {
        "core_claim": "선풍기를 틀고 자면 사망한다.",
        "evidence_ids": [1],
        "reason": "선풍기 사망설 문서는 이 속설에 과학적 근거가 없다고 설명한다.",
        "label": "반박",
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def _verify_raw(**overrides) -> str:
    data = {
        "core_claim": "선풍기를 틀고 자면 사망한다.",
        "refuting_sentence": "그러나 이 속설은 과학적 근거가 없다.",
        "refuting_ids": [1],
        "reason": "선풍기 사망설 문서는 이 속설에 과학적 근거가 없다고 설명한다.",
        "verdict": "반박",
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def test_schemas_put_verdict_last_so_model_reasons_before_judging():
    assert list(JUDGE_SCHEMA["properties"]) == ["core_claim", "evidence_ids", "reason", "label"]
    assert list(VERIFY_SCHEMA["properties"]) == [
        "core_claim", "refuting_sentence", "refuting_ids", "reason", "verdict",
    ]


def test_judge_prompt_numbers_blocks_and_carries_rules():
    prompt = build_judge_prompt(
        "선풍기를 틀고 자면 사망한다는 이야기를 들었어요.",
        [("선풍기 사망설", "과학적 근거가 없는 속설이다."), ("미신", "선풍기 미신이 있다.")],
    )

    assert "[1] (선풍기 사망설) 과학적 근거가 없는 속설이다." in prompt
    assert "[2] (미신) 선풍기 미신이 있다." in prompt
    assert "선풍기를 틀고 자면 사망한다는 이야기를 들었어요." in prompt
    assert "전해 들은" in prompt and "속설이나 틀린 이야기라고 소개" in prompt
    assert "거짓일 수밖에 없다" in prompt
    assert "소개만 하는 문단" in prompt and "\"미신\"" in prompt
    assert "{" not in prompt and "}" not in prompt


def test_verify_prompt_gets_original_sentence_and_only_given_blocks():
    prompt = build_verify_prompt(
        "선풍기를 틀고 자면 사망한다는 이야기를 들었어요.",
        [("선풍기 사망설", "그러나 이 속설은 과학적 근거가 없다.")],
    )

    assert "선풍기를 틀고 자면 사망한다는 이야기를 들었어요." in prompt
    assert "[1] (선풍기 사망설) 그러나 이 속설은 과학적 근거가 없다." in prompt
    assert "[2]" not in prompt
    assert "확실하지 않으면" in prompt
    assert "{" not in prompt and "}" not in prompt


def test_judge_prompt_keeps_braces_inside_evidence_text():
    prompt = build_judge_prompt("주장", [("수식", "집합 {1, 2}의 원소")])

    assert "집합 {1, 2}의 원소" in prompt


def test_parse_judge_accepts_valid_response():
    assert parse_judge_response(_judge_raw(evidence_ids=[2, 1]), evidence_count=3) == {
        "core_claim": "선풍기를 틀고 자면 사망한다.",
        "evidence_ids": [2, 1],
        "reason": "선풍기 사망설 문서는 이 속설에 과학적 근거가 없다고 설명한다.",
        "label": "반박",
    }


def test_parse_drops_out_of_range_duplicate_and_non_int_ids():
    assert parse_judge_response(_judge_raw(evidence_ids=[0, 1, 1, 4, "2", True, 3]), evidence_count=3)["evidence_ids"] == [1, 3]
    assert parse_verify_response(_verify_raw(refuting_ids=[2, 2, 5, False]), cited_count=2)["refuting_ids"] == [2]


def test_parse_judge_accepts_blank_reason_only_when_not_refutation():
    assert parse_judge_response(_judge_raw(label="판단불가", evidence_ids=[], reason=""), evidence_count=3)["label"] == "판단불가"
    assert parse_judge_response(_judge_raw(reason=""), evidence_count=3) is None


def test_parse_judge_rejects_malformed_responses():
    assert parse_judge_response(_judge_raw(label="모르겠음"), evidence_count=3) is None
    assert parse_judge_response(_judge_raw(core_claim=" "), evidence_count=3) is None
    assert parse_judge_response(_judge_raw(evidence_ids="1"), evidence_count=3) is None
    assert parse_judge_response('{"label": "반박", "reason": "이유"}', evidence_count=3) is None
    assert parse_judge_response("", evidence_count=3) is None
    assert parse_judge_response("[1, 2]", evidence_count=3) is None


def test_parse_verify_accepts_valid_and_non_refutation_without_reason():
    assert parse_verify_response(_verify_raw(), cited_count=1)["verdict"] == "반박"
    parsed = parse_verify_response(
        _verify_raw(verdict="반박 아님", refuting_sentence="", refuting_ids=[], reason=""), cited_count=1,
    )
    assert parsed["verdict"] == "반박 아님"


def test_parse_verify_rejects_malformed_responses():
    assert parse_verify_response(_verify_raw(verdict="반박임"), cited_count=1) is None
    assert parse_verify_response(_verify_raw(reason="  "), cited_count=1) is None
    assert parse_verify_response(_verify_raw(refuting_ids=None), cited_count=1) is None
    assert parse_verify_response("", cited_count=1) is None


def test_build_claim_uses_given_reason_and_blocks():
    claim = build_claim(
        "선풍기를 틀고 자면 사망한다.",
        "근거 없는 속설이다.",
        [("선풍기 사망설", "과학적 근거가 없는 속설이다.")],
    )

    assert claim == {
        "sentence": "선풍기를 틀고 자면 사망한다.",
        "reason": "근거 없는 속설이다.",
        "evidence": [
            {
                "title": "선풍기 사망설",
                "text": "과학적 근거가 없는 속설이다.",
                "url": "https://ko.wikipedia.org/wiki/선풍기_사망설",
            }
        ],
    }
