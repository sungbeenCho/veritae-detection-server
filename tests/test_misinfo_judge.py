"""judge_sentence의 판정 흐름: 1차 판정 → 반박 후보는 인용 문단만 보고 검증 → 검증이 고른 문단만 근거로.
Ollama와 e5 임베딩은 가짜로 바꾸고, 위키 인덱스는 임시 SQLite로 만든다."""
import json

import pytest
from kiwipiepy import Kiwi

import misinfo_infer
from misinfo_lib import JUDGE_SCHEMA, VERIFY_SCHEMA
from wiki_index import add_chunk, create_index, extract_keywords

SENTENCE = "선풍기를 틀고 자면 사망한다는 이야기를 들었어요."
FAN_DEATH_BLOCK = "선풍기를 켜고 자면 사망한다는 가설이 있다. 그러나 이 속설은 과학적 근거가 없다."


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


def _block_ids(prompt: str) -> dict[str, int]:
    """프롬프트의 근거 문단 번호를 제목별로 찾는다(번호는 검색 순위로 정해진다)."""
    lines = [line for line in prompt.splitlines() if line.startswith("[") and "] (" in line]
    return {line.split("] (", 1)[1].split(")", 1)[0]: i for i, line in enumerate(lines, start=1)}


def _judge(label, cite=("선풍기 사망설", "케니 맥코믹"), reason="1차 이유"):
    def respond(prompt):
        ids = _block_ids(prompt)
        return json.dumps(
            {"core_claim": "선풍기를 틀고 자면 사망한다.", "evidence_ids": [ids[t] for t in cite if t in ids],
             "reason": reason, "label": label},
            ensure_ascii=False,
        )
    return respond


def _verify(verdict, refute=("선풍기 사망설",), reason="선풍기 사망설 문서는 근거가 없다고 한다."):
    def respond(prompt):
        ids = _block_ids(prompt)
        return json.dumps(
            {"core_claim": "선풍기를 틀고 자면 사망한다.", "refuting_sentence": "그러나 이 속설은 과학적 근거가 없다.",
             "refuting_ids": [ids[t] for t in refute if t in ids], "reason": reason, "verdict": verdict},
            ensure_ascii=False,
        )
    return respond


@pytest.fixture
def ollama(monkeypatch):
    """ask_ollama를 가짜로 바꾼다. 1차/검증은 넘겨받은 응답 형식(schema)으로 구분한다."""
    calls = []
    responses = {}

    def fake_ask(url, model, prompt, schema):
        stage = "judge" if schema is JUDGE_SCHEMA else "verify" if schema is VERIFY_SCHEMA else "?"
        calls.append({"stage": stage, "prompt": prompt})
        response = responses[stage]
        return response(prompt) if callable(response) else response

    monkeypatch.setattr(misinfo_infer, "ask_ollama", fake_ask)
    monkeypatch.setattr(misinfo_infer, "rerank", lambda sentence, candidates, top_k: candidates[:top_k])
    return calls, responses


def _run(kiwi, conn):
    return misinfo_infer.judge_sentence(kiwi, conn, SENTENCE, "http://ollama", "qwen", evidence_count=5)


def test_non_refutation_is_not_verified(kiwi, conn, ollama):
    calls, responses = ollama
    responses["judge"] = _judge("판단불가", cite=())

    assert _run(kiwi, conn) is None
    assert [c["stage"] for c in calls] == ["judge"]


def test_refutation_without_cited_evidence_is_not_verified_or_shown(kiwi, conn, ollama):
    calls, responses = ollama
    responses["judge"] = _judge("반박", cite=())

    assert _run(kiwi, conn) is None
    assert [c["stage"] for c in calls] == ["judge"]


def test_verification_sees_original_sentence_and_only_cited_blocks(kiwi, conn, ollama):
    calls, responses = ollama
    responses["judge"] = _judge("반박", cite=("선풍기 사망설",))
    responses["verify"] = _verify("반박 아님", refute=())

    _run(kiwi, conn)

    verify_prompt = calls[1]["prompt"]
    assert SENTENCE in verify_prompt
    assert list(_block_ids(verify_prompt)) == ["선풍기 사망설"]


def test_rejected_verification_drops_the_refutation(kiwi, conn, ollama):
    _, responses = ollama
    responses["judge"] = _judge("반박")
    responses["verify"] = _verify("반박 아님", refute=())

    assert _run(kiwi, conn) is None


def test_verified_refutation_shows_only_blocks_verification_confirmed(kiwi, conn, ollama):
    """1차가 무관한 문단(케니 맥코믹)까지 인용해도, 검증이 실제로 반박한다고 고른 문단만 근거로 나간다."""
    _, responses = ollama
    responses["judge"] = _judge("반박", cite=("선풍기 사망설", "케니 맥코믹"))
    responses["verify"] = _verify("반박", refute=("선풍기 사망설",))

    claim = _run(kiwi, conn)

    assert claim == {
        "sentence": SENTENCE,
        "reason": "선풍기 사망설 문서는 근거가 없다고 한다.",
        "evidence": [
            {"title": "선풍기 사망설", "text": FAN_DEATH_BLOCK, "url": "https://ko.wikipedia.org/wiki/선풍기_사망설"}
        ],
    }


def test_verified_refutation_without_refuting_blocks_is_not_shown(kiwi, conn, ollama):
    _, responses = ollama
    responses["judge"] = _judge("반박")
    responses["verify"] = _verify("반박", refute=())

    assert _run(kiwi, conn) is None


def test_malformed_verification_is_not_shown(kiwi, conn, ollama):
    _, responses = ollama
    responses["judge"] = _judge("반박")
    responses["verify"] = ""

    assert _run(kiwi, conn) is None


def test_neighbor_context_is_included_in_judge_prompt(kiwi, conn, ollama):
    calls, responses = ollama
    responses["judge"] = _judge("판단불가", cite=())

    _run(kiwi, conn)

    assert FAN_DEATH_BLOCK in calls[0]["prompt"]


def test_ollama_request_fixes_context_size_and_disables_thinking(monkeypatch):
    sent = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"response": "{}"}).encode()

    def fake_urlopen(req, timeout):
        sent.update(json.loads(req.data))
        return FakeResponse()

    monkeypatch.setattr(misinfo_infer.urllib.request, "urlopen", fake_urlopen)

    misinfo_infer.ask_ollama("http://ollama", "qwen", "프롬프트", VERIFY_SCHEMA)

    assert sent["think"] is False
    assert sent["format"] == VERIFY_SCHEMA
    assert sent["options"]["num_ctx"] == misinfo_infer.OLLAMA_NUM_CTX
