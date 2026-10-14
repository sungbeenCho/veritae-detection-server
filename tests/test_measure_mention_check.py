"""'이미 맞는 말을 하는 문장' 확인 방식 측정 도구(scripts/measure_mention_check.py)의 테스트.

2026-10-14: LLM이 반박이라고 한 판정을 NLI가 위키 문장을 못 찾았다는 이유로 버리지 않게 바꾼다. 대신 LLM이
"한국에는 선풍기를 틀고 자면 죽는다는 속설이 있다"처럼 이미 맞는 말을 하는 문장을 거짓으로 착각한 경우만 막는다.
어느 방향(이유→문장 등)의 어떤 NLI 점수로 막을지는 데스크탑 NLI 모델로 재 보고 정한다."""
import pytest

from measure_mention_check import DIRECTIONS, best_rule, evaluate_rule, score_cases, summarize

SHOW = {"expected": "show", "sentence": "선풍기를 틀고 자면 사망한다는 이야기를 들었어요.",
        "core_claim": "선풍기를 틀고 자면 사망한다.", "reason": "선풍기 사망설은 근거 없는 미신이다."}
BLOCK = {"expected": "block", "sentence": "선풍기를 틀고 자면 죽는다는 건 미신이다.",
         "core_claim": "선풍기를 틀고 자면 죽는다는 건 미신이다.", "reason": "선풍기 사망설은 근거 없는 미신이다."}


def _probs(table):
    """(전제, 가설) -> 함의 확률. 나머지는 중립."""
    seen = []

    def probs(premises, hypothesis):
        seen.append((list(premises), hypothesis))
        return [{"entailment": table.get((p, hypothesis), 0.0), "neutral": 1 - table.get((p, hypothesis), 0.0),
                 "contradiction": 0.0} for p in premises]

    probs.seen = seen
    return probs


def test_score_cases_scores_every_direction_with_the_right_premise_and_hypothesis():
    probs = _probs({(SHOW["reason"], SHOW["sentence"]): 0.7, (SHOW["core_claim"], SHOW["reason"]): 0.2})

    [record] = score_cases([SHOW], probs)

    assert set(record["scores"]) == set(DIRECTIONS)
    assert record["scores"]["이유→문장"]["entailment"] == pytest.approx(0.7)
    assert record["scores"]["주장→이유"]["entailment"] == pytest.approx(0.2)
    assert record["scores"]["문장→이유"]["entailment"] == 0.0
    assert ([SHOW["sentence"]], SHOW["reason"]) in probs.seen
    assert record["sentence"] == SHOW["sentence"]


def _record(case, direction, entailment, contradiction=0.0):
    scores = {d: {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0} for d in DIRECTIONS}
    scores[direction] = {"entailment": entailment, "neutral": 1 - entailment - contradiction, "contradiction": contradiction}
    return {**case, "scores": scores}


def test_evaluate_rule_counts_blocked_mentions_and_wrongly_blocked_claims():
    records = [
        _record(BLOCK, "이유→문장", 0.9),
        _record({**BLOCK, "sentence": "다른 맞는 말"}, "이유→문장", 0.2),
        _record(SHOW, "이유→문장", 0.6),
        _record({**SHOW, "sentence": "다른 거짓 문장"}, "이유→문장", 0.1),
    ]

    result = evaluate_rule(records, "이유→문장", "entailment", 0.5)

    assert result["blocked_ok"] == 1
    assert result["block_total"] == 2
    assert result["wrongly_blocked"] == [SHOW["sentence"]]
    assert result["show_total"] == 2
    assert result["missed_blocks"] == ["다른 맞는 말"]


def test_evaluate_rule_threshold_is_inclusive():
    records = [_record(BLOCK, "문장→이유", 0.5)]

    assert evaluate_rule(records, "문장→이유", "entailment", 0.5)["blocked_ok"] == 1


def test_summarize_puts_rules_that_never_block_a_real_claim_first():
    """표시돼야 할 거짓 문장을 막지 않는 것이 먼저, 그다음 맞는 말을 많이 막는 순."""
    records = [_record(BLOCK, "이유→문장", 0.9), _record(SHOW, "이유→문장", 0.95), _record(SHOW, "문장→이유", 0.0)]

    rows = summarize(records)

    assert rows[0]["wrongly_blocked"] == []
    assert all(len(a["wrongly_blocked"]) <= len(b["wrongly_blocked"]) for a, b in zip(rows, rows[1:]))
    best_clean = [r for r in rows if not r["wrongly_blocked"]]
    assert all(a["blocked_ok"] >= b["blocked_ok"] for a, b in zip(best_clean, best_clean[1:]))


def _row(blocked_ok, wrongly_blocked):
    return {"blocked_ok": blocked_ok, "wrongly_blocked": ["거짓"] * wrongly_blocked}


def test_best_rule_is_the_one_blocking_most_mentions_without_blocking_any_claim():
    rows = [_row(0, 0), _row(5, 0), _row(9, 1)]

    assert best_rule(rows) == rows[1]


def test_best_rule_is_none_when_no_rule_blocks_anything_safely():
    """아무것도 막지 않는 조합을 '가장 나은 조합'으로 보여주면 안 된다."""
    assert best_rule([_row(0, 0), _row(7, 2)]) is None
