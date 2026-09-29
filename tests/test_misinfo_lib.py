from scripts.misinfo_lib import build_claim, build_prompt, parse_llm_response


def test_build_prompt_includes_sentence_and_evidence_titles():
    prompt = build_prompt(
        "선풍기를 틀고 자면 사망한다.",
        [("선풍기 사망설", "선풍기 사망설은 근거 없는 미신이다.")],
    )

    assert "선풍기를 틀고 자면 사망한다." in prompt
    assert "선풍기 사망설은 근거 없는 미신이다." in prompt
    assert "지지" in prompt and "반박" in prompt and "판단불가" in prompt


def test_parse_llm_response_accepts_valid_json():
    raw = '{"label": "반박", "reason": "근거 문단이 주장을 부정합니다."}'

    parsed = parse_llm_response(raw)

    assert parsed == {"label": "반박", "reason": "근거 문단이 주장을 부정합니다."}


def test_parse_llm_response_rejects_invalid_label():
    raw = '{"label": "모르겠음", "reason": "..."}'

    assert parse_llm_response(raw) is None


def test_parse_llm_response_rejects_missing_reason():
    raw = '{"label": "반박"}'

    assert parse_llm_response(raw) is None


def test_parse_llm_response_rejects_malformed_json():
    raw = "이건 JSON이 아니다"

    assert parse_llm_response(raw) is None


def test_build_claim_assembles_evidence_with_urls():
    verdict = {"label": "반박", "reason": "근거 문단이 주장을 부정합니다."}

    claim = build_claim(
        "선풍기를 틀고 자면 사망한다.",
        verdict,
        [("선풍기 사망설", "선풍기 사망설은 근거 없는 미신이다.")],
    )

    assert claim == {
        "sentence": "선풍기를 틀고 자면 사망한다.",
        "reason": "근거 문단이 주장을 부정합니다.",
        "evidence": [
            {
                "title": "선풍기 사망설",
                "text": "선풍기 사망설은 근거 없는 미신이다.",
                "url": "https://ko.wikipedia.org/wiki/선풍기_사망설",
            }
        ],
    }
