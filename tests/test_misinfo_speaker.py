"""--judge-mode speaker 판정 흐름: 4B 정리(위키 없이 내용만) → 말한 사람 판별 분류기 → 정리된 내용으로 검색·판정 → 문장이 틀렸으면
(맞다고 한 내용이 거짓, 아니라고 한 내용이 참) 근거 문장 하나. Ollama, 분류기, NLI는 가짜로 바꾸고 위키 인덱스는 임시 SQLite로 만든다."""
import json

import pytest
from kiwipiepy import Kiwi

import misinfo_infer
import misinfo_lib
from misinfo_timing import StepTimer
from wiki_index import add_chunk, create_index, extract_keywords

CLAIM = "선풍기를 틀고 자면 사망한다."
STATES = "선풍기를 틀고 자면 사망한다는 이야기를 들었어요."
DENIES = "선풍기를 틀고 자면 사망한다는 건 미신이래요."
INTRO = "선풍기를 켜고 자면 사망한다는 가설이 있다."
REFUTING = "그러나 이 속설은 과학적 근거가 없다."
SUPPORTING = "선풍기를 틀고 자면 사망할 수 있다는 연구가 있다."


@pytest.fixture(scope="module")
def kiwi():
    return Kiwi()


@pytest.fixture
def conn(tmp_path, kiwi):
    conn = create_index(tmp_path / "wiki.sqlite3", snapshot="2026-09-01")
    for text in (INTRO, REFUTING, SUPPORTING):
        add_chunk(conn, "선풍기 사망설", text, extract_keywords(kiwi, text))
    conn.commit()
    return conn


@pytest.fixture
def llm(monkeypatch):
    """정리 응답(claim)과 판정 응답(label)을 정한다. 스키마로 어느 호출인지 가른다."""
    calls = []
    answer = {"claim": CLAIM, "label": "반박", "reason": "선풍기 사망설은 근거가 없다고 한다.", "cite": [1]}

    def fake_ask(url, model, prompt, schema):
        calls.append((schema, prompt))
        if schema is misinfo_lib.EXTRACT_SCHEMA:
            return json.dumps({"claim": answer["claim"]}, ensure_ascii=False)
        return json.dumps({"evidence_ids": answer["cite"], "reason": answer["reason"], "label": answer["label"]},
                          ensure_ascii=False)

    monkeypatch.setattr(misinfo_infer, "ask_ollama", fake_ask)
    monkeypatch.setattr(misinfo_infer, "rerank", lambda sentence, candidates, top_k: candidates[:top_k])
    return calls, answer


class FakeSpeaker:
    """원문마다 정해진 NLI 라벨(entailment = 맞다고 함, contradiction = 아니라고 함, neutral = 사실 없음)."""

    def __init__(self, labels: dict[str, str]):
        self.labels = labels
        self.seen = []

    def label_probabilities(self, premises, hypothesis):
        self.seen.append((list(premises), hypothesis))
        return [{name: (0.9 if name == self.labels[p] else 0.05) for name in ("entailment", "neutral", "contradiction")}
                for p in premises]


def _scorer(scores: dict[str, float]):
    seen = []

    def score(premises, hypothesis):
        seen.append((list(premises), hypothesis))
        return [scores.get(p, 0.01) for p in premises]

    score.seen = seen
    return score


def _run(kiwi, conn, sentence, speaker_label, contradiction=None, entailment=None):
    timer = StepTimer()
    contradiction = contradiction or _scorer({})
    entailment = entailment or _scorer({})
    result = misinfo_infer.judge_sentence_speaker(
        kiwi, conn, sentence, "u", "m", 5, contradiction, entailment, FakeSpeaker({sentence: speaker_label}), timer=timer,
    )
    return result, timer, contradiction, entailment


def test_stated_false_content_is_shown_with_contradicting_sentence(kiwi, conn, llm):
    _, answer = llm
    answer.update(label="반박")

    result, timer, contradiction, entailment = _run(kiwi, conn, STATES, "entailment", contradiction=_scorer({REFUTING: 0.95}))

    assert result["sentence"] == STATES
    assert result["evidence"][0]["text"] == REFUTING
    assert timer.outcome == "refuted"
    assert contradiction.seen and not entailment.seen


def test_denied_false_content_is_not_shown(kiwi, conn, llm):
    """"…죽는다는 건 미신이래요"는 "죽는다"가 거짓이면 맞는 문장이다(2026-10-14 사용자 결정)."""
    _, answer = llm
    answer.update(label="반박")

    result, timer, _, _ = _run(kiwi, conn, DENIES, "contradiction", contradiction=_scorer({REFUTING: 0.95}))

    assert result is None
    assert timer.outcome == "not_refuted"


def test_denied_true_content_is_shown_with_supporting_sentence(kiwi, conn, llm):
    """참인 내용을 아니라고 한 문장은 틀린 문장이다. 근거는 그 내용을 뒷받침하는(함의) 문장이다."""
    _, answer = llm
    answer.update(label="지지", reason="선풍기 사망설 문서는 사망할 수 있다고 한다.")

    result, timer, contradiction, entailment = _run(kiwi, conn, DENIES, "contradiction", entailment=_scorer({SUPPORTING: 0.9}))

    assert result["evidence"][0]["text"] == SUPPORTING
    assert entailment.seen and not contradiction.seen


def test_stated_true_content_is_not_shown(kiwi, conn, llm):
    _, answer = llm
    answer.update(label="지지")

    result, timer, _, _ = _run(kiwi, conn, STATES, "entailment")

    assert result is None


def test_speaker_no_fact_stops_before_search_and_judgment(kiwi, conn, llm):
    calls, _ = llm

    result, timer, _, _ = _run(kiwi, conn, STATES, "neutral")

    assert result is None
    assert timer.outcome == "no_fact_speaker"
    assert [schema for schema, _ in calls] == [misinfo_lib.EXTRACT_SCHEMA]


def test_empty_claim_is_no_fact_without_classifier(kiwi, conn, llm):
    calls, answer = llm
    answer.update(claim="")
    speaker = FakeSpeaker({})

    timer = StepTimer()
    result = misinfo_infer.judge_sentence_speaker(kiwi, conn, "내일 점심 같이 먹을래요?", "u", "m", 5, _scorer({}), _scorer({}),
                                                  speaker, timer=timer)

    assert result is None and timer.outcome == "no_fact" and speaker.seen == []


def test_invented_words_fall_back_to_cannot_judge(kiwi, conn, llm):
    """원문에 없는 명사가 정리에 들어가면(지어냄) 판정하지 않는다."""
    calls, answer = llm
    answer.update(claim="선풍기를 틀고 자면 뇌졸중이 예방된다.")

    result, timer, _, _ = _run(kiwi, conn, STATES, "entailment")

    assert result is None and timer.outcome == "extract_unfaithful"
    assert len(calls) == 1


def test_classifier_reads_original_sentence_and_claim(kiwi, conn, llm):
    _, answer = llm
    answer.update(label="판단불가")
    speaker = FakeSpeaker({STATES: "entailment"})

    misinfo_infer.judge_sentence_speaker(kiwi, conn, STATES, "u", "m", 5, _scorer({}), _scorer({}), speaker)

    assert speaker.seen == [([STATES], CLAIM)]


def test_judgment_uses_claim_and_one_paragraph_template(kiwi, conn, llm):
    calls, answer = llm
    answer.update(label="판단불가")

    _run(kiwi, conn, STATES, "entailment")

    judge_prompt = calls[1][1]
    assert judge_prompt.rstrip().endswith(CLAIM)
    assert "가장 직접 알려주는 근거 문단 하나의 번호만" in judge_prompt
    assert "core_claim" not in judge_prompt


def test_templates_a_differ_from_current_only_in_paragraph_rules():
    assert misinfo_lib.JUDGE_TEMPLATE_A != misinfo_lib.JUDGE_TEMPLATE
    assert "가장 직접 알려주는 근거 문단 하나의 번호만" in misinfo_lib.JUDGE_TEMPLATE_A
    assert "다른 문단의 내용은 섞지 않는다." in misinfo_lib.CLAIM_JUDGE_TEMPLATE_A
    assert "번호를 모두 적는다" not in misinfo_lib.CLAIM_JUDGE_TEMPLATE_A


def test_parse_extract_response():
    assert misinfo_lib.parse_extract_response('{"claim": " 지구는 둥글다. "}') == "지구는 둥글다."
    assert misinfo_lib.parse_extract_response('{"claim": ""}') == ""
    assert misinfo_lib.parse_extract_response('{"stance": "전함"}') is None
    assert misinfo_lib.parse_extract_response("깨짐") is None


def test_sentence_is_wrong_rule():
    S, D, N = misinfo_lib.SPEAKER_STATES, misinfo_lib.SPEAKER_DENIES, misinfo_lib.SPEAKER_NO_FACT
    assert misinfo_lib.sentence_is_wrong(S, "반박") and misinfo_lib.sentence_is_wrong(D, "지지")
    assert not misinfo_lib.sentence_is_wrong(S, "지지") and not misinfo_lib.sentence_is_wrong(D, "반박")
    assert not misinfo_lib.sentence_is_wrong(N, "반박") and not misinfo_lib.sentence_is_wrong(S, "판단불가")
