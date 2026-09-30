"""judge_sentence의 판정 흐름: 1차 판정 → 반박 후보만 생각 모드로 정밀 재판정 → 인용된 근거만 표시.
Ollama와 e5 임베딩은 가짜로 바꾸고, 위키 인덱스는 임시 SQLite로 만든다."""
import json

import pytest
from kiwipiepy import Kiwi

import misinfo_infer
from wiki_index import add_chunk, create_index, extract_keywords

SENTENCE = "선풍기를 틀고 자면 사망한다는 이야기를 들었어요."


@pytest.fixture(scope="module")
def kiwi():
    return Kiwi()


@pytest.fixture
def conn(tmp_path, kiwi):
    conn = create_index(tmp_path / "wiki.sqlite3", snapshot="2026-09-01")
    for title, text in [
        ("케니 맥코믹", "선풍기에 말려서 사망하는 장면이 나온다."),
        ("선풍기 사망설", "선풍기를 켜고 자면 사망한다는 가설이 있다."),
        ("선풍기 사망설", "그러나 이 속설은 과학적 근거가 없다."),
    ]:
        add_chunk(conn, title, text, extract_keywords(kiwi, text))
    conn.commit()
    return conn


def _response(label, evidence_ids=(1,), reason="이유"):
    return json.dumps(
        {"core_claim": "선풍기를 틀고 자면 사망한다.", "evidence_ids": list(evidence_ids), "reason": reason, "label": label},
        ensure_ascii=False,
    )


@pytest.fixture
def ollama(monkeypatch):
    """ask_ollama를 가짜로 바꾸고, think 값별 응답과 호출 기록을 돌려준다."""
    calls = []
    responses = {}

    def fake_ask(url, model, prompt, think):
        calls.append({"think": think, "prompt": prompt})
        response = responses[think]
        return response(prompt) if callable(response) else response

    monkeypatch.setattr(misinfo_infer, "ask_ollama", fake_ask)
    monkeypatch.setattr(misinfo_infer, "rerank", lambda sentence, candidates, top_k: candidates[:top_k])
    return calls, responses


def _judge(kiwi, conn):
    return misinfo_infer.judge_sentence(kiwi, conn, SENTENCE, "http://ollama", "qwen", evidence_count=5)


def test_non_refutation_is_not_rechecked(kiwi, conn, ollama):
    calls, responses = ollama
    responses[False] = _response("판단불가", evidence_ids=[])

    assert _judge(kiwi, conn) is None
    assert [c["think"] for c in calls] == [False]


def test_refutation_is_rechecked_with_thinking_and_final_verdict_decides(kiwi, conn, ollama):
    calls, responses = ollama
    responses[False] = _response("반박")
    responses[True] = _response("지지")

    assert _judge(kiwi, conn) is None
    assert [c["think"] for c in calls] == [False, True]
    assert calls[0]["prompt"] == calls[1]["prompt"]


def _cite_fan_death_block(prompt):
    """프롬프트에서 '선풍기 사망설' 문단의 번호를 찾아 그것만 인용한다(번호는 검색 순위로 정해진다)."""
    block_lines = [line for line in prompt.splitlines() if line.startswith("[") and "] (" in line]
    fan_death_id = next(i for i, line in enumerate(block_lines, start=1) if "(선풍기 사망설)" in line)
    return _response("반박", evidence_ids=[fan_death_id], reason="선풍기 사망설 문서는 근거가 없다고 한다.")


def test_confirmed_refutation_shows_only_evidence_cited_by_final_verdict(kiwi, conn, ollama):
    _, responses = ollama
    responses[False] = _response("반박", evidence_ids=[1, 2])
    responses[True] = _cite_fan_death_block

    claim = _judge(kiwi, conn)

    assert claim["reason"] == "선풍기 사망설 문서는 근거가 없다고 한다."
    assert [e["title"] for e in claim["evidence"]] == ["선풍기 사망설"]
    assert claim["evidence"][0]["text"] == "선풍기를 켜고 자면 사망한다는 가설이 있다. 그러나 이 속설은 과학적 근거가 없다."


def test_refutation_without_any_cited_evidence_is_not_shown(kiwi, conn, ollama):
    _, responses = ollama
    responses[False] = _response("반박")
    responses[True] = _response("반박", evidence_ids=[])

    assert _judge(kiwi, conn) is None


def test_malformed_recheck_response_is_not_shown(kiwi, conn, ollama):
    _, responses = ollama
    responses[False] = _response("반박")
    responses[True] = "JSON이 아님"

    assert _judge(kiwi, conn) is None


def test_neighbor_context_is_included_in_prompt(kiwi, conn, ollama):
    calls, responses = ollama
    responses[False] = _response("판단불가", evidence_ids=[])

    _judge(kiwi, conn)

    assert "선풍기를 켜고 자면 사망한다는 가설이 있다. 그러나 이 속설은 과학적 근거가 없다." in calls[0]["prompt"]
