import json

from scripts.misinfo_lib import OLLAMA_SCHEMA, build_claim, build_prompt, parse_llm_response


def _raw(**overrides) -> str:
    data = {
        "core_claim": "선풍기를 틀고 자면 사망한다.",
        "evidence_ids": [1],
        "reason": "선풍기 사망설 문서는 이 속설에 과학적 근거가 없다고 설명한다.",
        "label": "반박",
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def test_schema_puts_label_last_so_model_reasons_before_judging():
    assert list(OLLAMA_SCHEMA["properties"]) == ["core_claim", "evidence_ids", "reason", "label"]


def test_build_prompt_numbers_evidence_blocks_and_includes_sentence():
    prompt = build_prompt(
        "선풍기를 틀고 자면 사망한다는 이야기를 들었어요.",
        [("선풍기 사망설", "과학적 근거가 없는 속설이다."), ("미신", "선풍기 미신이 있다.")],
    )

    assert "[1] (선풍기 사망설) 과학적 근거가 없는 속설이다." in prompt
    assert "[2] (미신) 선풍기 미신이 있다." in prompt
    assert "선풍기를 틀고 자면 사망한다는 이야기를 들었어요." in prompt
    assert "전해 들은" in prompt


def test_parse_accepts_valid_response():
    parsed = parse_llm_response(_raw(evidence_ids=[2, 1]), evidence_count=3)

    assert parsed == {
        "core_claim": "선풍기를 틀고 자면 사망한다.",
        "evidence_ids": [2, 1],
        "reason": "선풍기 사망설 문서는 이 속설에 과학적 근거가 없다고 설명한다.",
        "label": "반박",
    }


def test_parse_drops_out_of_range_duplicate_and_non_int_ids():
    parsed = parse_llm_response(_raw(evidence_ids=[0, 1, 1, 4, "2", True, 3]), evidence_count=3)

    assert parsed["evidence_ids"] == [1, 3]


def test_parse_rejects_invalid_label():
    assert parse_llm_response(_raw(label="모르겠음"), evidence_count=3) is None


def test_parse_accepts_blank_reason_only_when_not_refutation():
    assert parse_llm_response(_raw(label="판단불가", evidence_ids=[], reason=""), evidence_count=3)["label"] == "판단불가"


def test_parse_rejects_missing_or_blank_fields():
    assert parse_llm_response(_raw(reason=""), evidence_count=3) is None
    assert parse_llm_response(_raw(core_claim=" "), evidence_count=3) is None
    assert parse_llm_response(_raw(evidence_ids="1"), evidence_count=3) is None
    assert parse_llm_response('{"label": "반박", "reason": "이유"}', evidence_count=3) is None


def test_parse_rejects_malformed_json():
    assert parse_llm_response("이건 JSON이 아니다", evidence_count=3) is None
    assert parse_llm_response("[1, 2]", evidence_count=3) is None


def test_build_claim_includes_only_cited_evidence():
    blocks = [
        ("케니 맥코믹", "선풍기에 말려서 사망."),
        ("선풍기 사망설", "과학적 근거가 없는 속설이다."),
        ("미신", "선풍기 미신이 있다."),
    ]
    verdict = {"core_claim": "c", "evidence_ids": [2], "reason": "근거 없는 속설이다.", "label": "반박"}

    claim = build_claim("선풍기를 틀고 자면 사망한다.", verdict, blocks)

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
