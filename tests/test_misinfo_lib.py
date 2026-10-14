import json

from scripts.misinfo_lib import (
    JUDGE_SCHEMA,
    PREMISE_MAX_CHARS,
    build_claim,
    build_judge_prompt,
    fit_blocks,
    parse_judge_response,
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

    prompt = build_judge_prompt(sentence, worst)
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


def test_schema_puts_label_last_so_model_reasons_before_judging():
    assert list(JUDGE_SCHEMA["properties"]) == ["core_claim", "evidence_ids", "reason", "label"]


def test_judge_prompt_numbers_blocks_and_carries_rules():
    prompt = build_judge_prompt(
        "선풍기를 틀고 자면 사망한다는 이야기를 들었어요.",
        [("선풍기 사망설", "과학적 근거가 없는 속설이다."), ("미신", "선풍기 미신이 있다.")],
    )

    assert "[1] (선풍기 사망설) 과학적 근거가 없는 속설이다." in prompt
    assert "[2] (미신) 선풍기 미신이 있다." in prompt
    assert "선풍기를 틀고 자면 사망한다는 이야기를 들었어요." in prompt
    assert "거짓일 수밖에 없다" in prompt
    assert "소개만 하는 문단" in prompt and "\"미신\"" in prompt
    assert "판정 예시" in prompt and "이름을 정확히 비교" in prompt
    assert "{" not in prompt and "}" not in prompt


def test_core_claim_rule_states_what_the_sentence_finally_says_about_the_world():
    """2026-10-14 사용자 결정: 문장이 결국 무엇을 말하는지로 판단한다.
    - 전해 들은 말("X라고 들었어요")과 속설 소개("X라는 속설이 있다")는 X 자체를 판단한다.
    - 부정하는 문장("X는 사실이 아니다")은 부정된 뜻(X가 아니다)을 판단한다.
    예전 규칙("속설을 소개하는 문장은 문장 그대로" = 속설이 있다는 사실을 판단)은 사용자 결정이 아니었고, 깨진 OCR
    문장("…이야기틀 들없어요")을 "…라는 속설이 있다"로 정리해 엉뚱한 것을 판단하게 만들었다(이미지 실측)."""
    prompt = build_judge_prompt("주장", [])

    assert "결국 무엇을 말하는지" in prompt
    assert "전해 들은 내용 자체" in prompt
    assert "\"~라는 속설이 있다\"" in prompt and "그 이야기의 내용 자체" in prompt
    assert "\"~는 사실이 아니다\"" in prompt and "부정된 뜻" in prompt
    assert "문장 그대로를 적는다" not in prompt
    assert "들없어요" in prompt  # OCR 오타로 깨진 전해 들은 말


def test_examples_do_not_leak_regression_test_sentences():
    """예시가 회귀 테스트 문장을 담으면 시험 답을 가르치는 셈이라 실측이 무의미해진다. 판정 예시뿐 아니라
    core_claim 규칙 안의 예시까지 프롬프트 전체(근거 문단과 주장 자리를 비운 상태)를 본다."""
    subjects = [
        "선풍기", "만리장성", "백신", "에펠탑", "세종", "한글", "부산", "서울", "태양", "지구", "아폴로", "뇌",
        "에베레스트", "이순신", "커피", "훈민정음", "신라", "고려", "독일", "영국", "런던", "금붕어", "번개",
        "고래", "피카소", "베토벤", "후지산", "토마토", "광화문", "임진왜란", "달 착륙", "달빛",
    ]
    prompt = build_judge_prompt("주장", [])
    assert all(subject not in prompt for subject in subjects)


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
