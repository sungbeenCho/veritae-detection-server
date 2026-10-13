"""'NLI 먼저' 구조 측정 도구(scripts/measure_nli_gate.py)의 테스트.

가짜정보 판정은 지금 모든 문장에 LLM을 먼저 돌려 문장당 9~19초가 든다(2026-10-13: 이미지 문장 2개가 2분).
LLM 호출을 줄이는 방법으로, 모든 문장을 위키 검색 + NLI(빠른 모델)로 먼저 검사하고 거짓 후보가 나온 문장에만
LLM을 돌리는 구조를 검토한다. 그 구조가 거짓 문장을 얼마나 놓치는지 재기 전에는 도입할 수 없다."""
import pytest
from kiwipiepy import Kiwi

import misinfo_infer
from measure_nli_gate import gate_scores, summarize
from wiki_index import add_chunk, create_index, extract_keywords

SENTENCE = "선풍기를 틀고 자면 사망한다는 이야기를 들었어요."
REFUTING = "그러나 이 속설은 과학적 근거가 없다."
UNRELATED = "선풍기에 말려서 사망하는 장면이 나온다."
INTRO = "선풍기를 켜고 자면 사망한다는 가설이 있다."


@pytest.fixture(scope="module")
def kiwi():
    return Kiwi()


@pytest.fixture
def conn(tmp_path, kiwi):
    conn = create_index(tmp_path / "wiki.sqlite3", snapshot="2026-09-01")
    # 반박 문장은 키워드가 없어 직접 검색되지 않고, 실서버처럼 같은 문서의 앞 조각에 딸려 온다(앞뒤 문맥 붙이기).
    for title, text in [("케니 맥코믹", UNRELATED), ("선풍기 사망설", INTRO), ("선풍기 사망설", REFUTING)]:
        add_chunk(conn, title, text, extract_keywords(kiwi, text))
    conn.commit()
    return conn


def _nli(scores_by_sentence):
    seen = []

    def score(premises, hypothesis):
        seen.append((list(premises), hypothesis))
        return [scores_by_sentence.get(p, 0.01) for p in premises]

    score.seen = seen
    return score


def test_gate_scores_uses_raw_sentence_as_hypothesis_and_returns_best_candidate(kiwi, conn, monkeypatch):
    """LLM이 정리한 핵심 주장이 없는 구조이므로 NLI에는 문장 원문을 그대로 넣는다."""
    monkeypatch.setattr(misinfo_infer, "rerank", lambda sentence, candidates, top_k: candidates[:top_k])
    nli = _nli({REFUTING: 0.93, UNRELATED: 0.2})

    result = gate_scores(kiwi, conn, SENTENCE, 5, nli)

    assert result["max_score"] == pytest.approx(0.93)
    assert result["best_title"] == "선풍기 사망설"
    assert result["best_sentence"] == REFUTING
    assert result["candidate_count"] == 3  # 무관한 문장, 도입 문장, 반박 문장
    assert nli.seen[0][1] == SENTENCE


def test_gate_scores_is_zero_without_candidates_and_does_not_call_nli(kiwi, conn):
    nli = _nli({})

    result = gate_scores(kiwi, conn, "!!!", 5, nli)

    assert result["max_score"] == 0.0
    assert result["best_sentence"] is None
    assert nli.seen == []


def _record(expected, category, max_score):
    return {"expected": expected, "category": category, "sentence": f"{category}-{max_score}", "max_score": max_score}


def test_summarize_counts_recall_gate_pass_and_savings_per_threshold():
    records = [
        _record("contradiction", "hearsay", 0.95),
        _record("contradiction", "hearsay", 0.30),
        _record("contradiction", "e2e", 0.85),
        _record("entailment", "mention", 0.90),
        _record("entailment", "e2e", 0.10),
        _record("entailment", "e2e", 0.05),
    ]

    summary = summarize(records, thresholds=[0.5, 0.8])

    at_half = summary[0.5]
    assert at_half["recall"] == pytest.approx(2 / 3)
    assert at_half["missed"] == 1
    assert at_half["false_gate_pass"] == 1  # 거짓이 아닌 문장이 LLM까지 가는 경우(LLM이 거른다) 1건
    assert at_half["llm_skipped"] == 3  # 6문장 중 0.5 미만 3건은 LLM을 건너뜀
    assert at_half["llm_skipped_rate"] == pytest.approx(0.5)
    at_eight = summary[0.8]
    assert at_eight["recall"] == pytest.approx(2 / 3)
    assert at_eight["missed_sentences"] == ["hearsay-0.3"]


def test_summarize_reports_recall_by_category():
    records = [
        _record("contradiction", "hearsay", 0.95),
        _record("contradiction", "hearsay", 0.30),
        _record("contradiction", "e2e", 0.85),
    ]

    by_category = summarize(records, thresholds=[0.8])[0.8]["recall_by_category"]

    assert by_category == {"hearsay": pytest.approx(0.5), "e2e": pytest.approx(1.0)}


def test_summarize_handles_no_contradiction_cases():
    summary = summarize([_record("entailment", "e2e", 0.1)], thresholds=[0.8])

    assert summary[0.8]["recall"] is None
