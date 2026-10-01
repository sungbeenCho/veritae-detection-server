"""judge_sentence의 판정 흐름: LLM 1차 판정 → 반박이면 검색 문단의 문장을 e5로 추리고 그중 근거 문장을 LLM이 고름 →
고른 문장만 NLI로 모순 확인 → 확인된 문장만 근거로.
Ollama, e5 임베딩, NLI 모델은 가짜로 바꾸고, 위키 인덱스는 임시 SQLite로 만든다."""
import json

import pytest
from kiwipiepy import Kiwi

import misinfo_infer
from wiki_index import add_chunk, create_index, extract_keywords

SENTENCE = "선풍기를 틀고 자면 사망한다는 이야기를 들었어요."
CLAIM = "선풍기를 틀고 자면 사망한다."
REFUTING = "그러나 이 속설은 과학적 근거가 없다."
FAN_DEATH_BLOCK = f"선풍기를 켜고 자면 사망한다는 가설이 있다. {REFUTING}"


@pytest.fixture(scope="module")
def kiwi():
    return Kiwi()


@pytest.fixture
def conn(tmp_path, kiwi):
    conn = create_index(tmp_path / "wiki.sqlite3", snapshot="2026-09-01")
    for title, text in [
        ("케니 맥코믹", "선풍기에 말려서 사망하는 장면이 나온다."),
        ("선풍기 사망설", "선풍기를 켜고 자면 사망한다는 가설이 있다."),
        ("선풍기 사망설", REFUTING),
    ]:
        add_chunk(conn, title, text, extract_keywords(kiwi, text))
    conn.commit()
    return conn


def _block_ids(prompt: str) -> dict[str, int]:
    lines = [line for line in prompt.splitlines() if line.startswith("[") and "] (" in line]
    return {line.split("] (", 1)[1].split(")", 1)[0]: i for i, line in enumerate(lines, start=1)}


@pytest.fixture
def llm(monkeypatch):
    """LLM 응답을 정한다. 1차 판정에서 인용할 문단은 제목으로(번호는 검색 순위로 정해진다),
    근거 문장 선택에서 고를 문장은 문장 원문으로 지정한다."""
    calls = []
    answer = {}

    def fake_ask(url, model, prompt, schema):
        calls.append((schema, prompt))
        if schema is misinfo_infer.SELECT_SCHEMA:
            ids = _sentence_ids(prompt)
            return json.dumps(
                {"analysis": "분석", "sentence_ids": [ids[s] for s in answer.get("pick", ()) if s in ids]},
                ensure_ascii=False,
            )
        ids = _block_ids(prompt)
        return json.dumps(
            {"core_claim": CLAIM, "evidence_ids": [ids[t] for t in answer["cite"] if t in ids],
             "reason": answer.get("reason", "[1] 문서는 근거가 없다고 한다."), "label": answer["label"]},
            ensure_ascii=False,
        )

    monkeypatch.setattr(misinfo_infer, "ask_ollama", fake_ask)
    monkeypatch.setattr(misinfo_infer, "rerank", lambda sentence, candidates, top_k: candidates[:top_k])
    return calls, answer


def _sentence_ids(prompt: str) -> dict[str, int]:
    """근거 문장 선택 프롬프트의 "[i] (제목) 문장" 줄에서 문장 → 번호."""
    lines = [line for line in prompt.splitlines() if line.startswith("[") and "] (" in line]
    return {line.split(") ", 1)[1]: int(line[1:line.index("]")]) for line in lines}


def _prompts(calls, schema) -> list[str]:
    return [prompt for s, prompt in calls if s is schema]


def _nli(contradicting: set[str]):
    """주어진 문장에만 모순 확률 0.99, 나머지는 0.01을 주는 가짜 NLI. 호출 기록도 남긴다."""
    seen = []

    def score(premises, hypothesis):
        seen.append((list(premises), hypothesis))
        return [0.99 if p in contradicting else 0.01 for p in premises]

    score.seen = seen
    return score


def _run(kiwi, conn, nli):
    return misinfo_infer.judge_sentence(kiwi, conn, SENTENCE, "http://ollama", "qwen", 5, nli)


def test_non_refutation_skips_selection_and_nli(kiwi, conn, llm):
    calls, answer = llm
    answer.update(label="판단불가", cite=())
    nli = _nli({REFUTING})

    assert _run(kiwi, conn, nli) is None
    assert _prompts(calls, misinfo_infer.SELECT_SCHEMA) == []
    assert nli.seen == []


def test_refutation_without_cited_evidence_is_not_shown(kiwi, conn, llm):
    calls, answer = llm
    answer.update(label="반박", cite=())
    nli = _nli({REFUTING})

    assert _run(kiwi, conn, nli) is None
    assert _prompts(calls, misinfo_infer.SELECT_SCHEMA) == []
    assert nli.seen == []


def test_selection_sees_sentences_of_all_retrieved_blocks_against_core_claim(kiwi, conn, llm):
    """1차 판정이 엉뚱한 문단을 인용해도 맞는 근거를 놓치지 않도록, 검색된 문단 전체가 후보다."""
    calls, answer = llm
    answer.update(label="반박", cite=("케니 맥코믹",), pick=(REFUTING,))

    claim = _run(kiwi, conn, _nli({REFUTING}))

    [select_prompt] = _prompts(calls, misinfo_infer.SELECT_SCHEMA)
    assert set(_sentence_ids(select_prompt)) == {
        "선풍기를 켜고 자면 사망한다는 가설이 있다.", REFUTING, "선풍기에 말려서 사망하는 장면이 나온다.",
    }
    assert CLAIM in select_prompt
    assert SENTENCE not in select_prompt
    assert claim["evidence"][0]["text"] == REFUTING


def test_shortlist_keeps_sentences_closest_to_claim(monkeypatch):
    import torch

    vectors = {"query: 주장": [1.0, 0.0], "passage: 가까움": [0.9, 0.1], "passage: 중간": [0.5, 0.5], "passage: 멂": [0.0, 1.0]}
    monkeypatch.setattr(misinfo_infer, "embed", lambda texts: torch.tensor([vectors[t] for t in texts]))
    candidates = [("A", "멂"), ("B", "중간"), ("C", "가까움")]

    assert misinfo_infer.shortlist_sentences("주장", candidates, size=2) == [("C", "가까움"), ("B", "중간")]


def test_shortlist_returns_all_when_few_candidates(monkeypatch):
    monkeypatch.setattr(misinfo_infer, "embed", lambda texts: (_ for _ in ()).throw(AssertionError("호출되면 안 된다")))

    assert misinfo_infer.shortlist_sentences("주장", [("A", "문장")], size=5) == [("A", "문장")]


def test_nli_checks_only_selected_sentences(kiwi, conn, llm):
    """NLI는 주어만 같은 무관한 문장에도 높은 모순 점수를 줄 수 있어, LLM이 고른 문장만 확인한다."""
    _, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",), pick=(REFUTING,))
    nli = _nli({REFUTING})

    _run(kiwi, conn, nli)

    assert nli.seen == [([REFUTING], CLAIM)]


def test_unselected_sentence_is_not_shown_even_if_nli_calls_it_contradiction(kiwi, conn, llm):
    _, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설", "케니 맥코믹"), pick=())

    assert _run(kiwi, conn, _nli({REFUTING, "선풍기에 말려서 사망하는 장면이 나온다."})) is None


def test_selected_sentence_not_confirmed_by_nli_is_not_shown(kiwi, conn, llm):
    _, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",), pick=("선풍기를 켜고 자면 사망한다는 가설이 있다.",))

    assert _run(kiwi, conn, _nli({REFUTING})) is None


def test_malformed_selection_response_is_not_shown(kiwi, conn, llm, monkeypatch):
    _, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",))
    real_fake = misinfo_infer.ask_ollama

    def broken_select(url, model, prompt, schema):
        return "형식 깨짐" if schema is misinfo_infer.SELECT_SCHEMA else real_fake(url, model, prompt, schema)

    monkeypatch.setattr(misinfo_infer, "ask_ollama", broken_select)

    assert _run(kiwi, conn, _nli({REFUTING})) is None


def test_confirmed_refutation_shows_only_the_refuting_sentence_and_link(kiwi, conn, llm):
    """1차가 무관한 문단(케니 맥코믹)까지 인용해도 고르고 확인된 반박 문장 하나와 링크만 나간다."""
    _, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설", "케니 맥코믹"), pick=(REFUTING,))

    claim = _run(kiwi, conn, _nli({REFUTING}))

    assert claim["sentence"] == SENTENCE
    assert [e["title"] for e in claim["evidence"]] == ["선풍기 사망설"]
    assert claim["evidence"][0]["text"] == REFUTING
    assert claim["evidence"][0]["url"] == "https://ko.wikipedia.org/wiki/선풍기_사망설"


def test_block_numbers_in_reason_become_titles(kiwi, conn, llm):
    calls, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",), pick=(REFUTING,))

    claim = _run(kiwi, conn, _nli({REFUTING}))

    first_title = next(iter(_block_ids(_prompts(calls, misinfo_infer.JUDGE_SCHEMA)[0])))
    assert claim["reason"] == f"'{first_title}' 문서는 근거가 없다고 한다."


def test_neighbor_context_is_included_in_judge_prompt(kiwi, conn, llm):
    calls, answer = llm
    answer.update(label="판단불가", cite=())

    _run(kiwi, conn, _nli(set()))

    assert FAN_DEATH_BLOCK in _prompts(calls, misinfo_infer.JUDGE_SCHEMA)[0]


def test_wiki_search_adds_articles_local_keyword_search_missed(kiwi, tmp_path, llm, monkeypatch):
    """로컬 키워드 검색은 단어가 다르면 못 찾는다("조작" vs "음모론") - 위키백과 검색이 찾은 문서를 후보에 더한다."""
    conn = create_index(tmp_path / "w.sqlite3", snapshot="2026-09-01")
    for title, text in [("아폴로 11호", "아폴로 11호는 1969년 달에 착륙했다."),
                        ("달착륙 음모론", "달착륙 음모론은 아폴로 계획이 날조되었다는 주장이다. 과학자들은 이를 반박했다.")]:
        add_chunk(conn, title, text, extract_keywords(kiwi, text))
    conn.commit()
    sent = []
    monkeypatch.setattr(misinfo_infer, "wiki_search_titles", lambda terms: sent.append(terms) or ["달착륙 음모론"])
    calls, answer = llm
    answer.update(label="판단불가", cite=())

    misinfo_infer.judge_sentence(kiwi, conn, "아폴로 11호의 달 착륙은 조작되었다.", "u", "m", 5, _nli(set()), wiki_search=True)

    judge_prompt = _prompts(calls, misinfo_infer.JUDGE_SCHEMA)[0]
    assert "(달착륙 음모론)" in judge_prompt
    assert sent == ["아폴로 11 달 착륙 조작"]  # 문장이 아니라 핵심 단어만 보낸다


def test_wiki_search_is_not_used_unless_enabled(kiwi, conn, llm, monkeypatch):
    monkeypatch.setattr(misinfo_infer, "wiki_search_titles", lambda terms: (_ for _ in ()).throw(AssertionError("호출되면 안 된다")))
    _, answer = llm
    answer.update(label="판단불가", cite=())

    assert _run(kiwi, conn, _nli(set())) is None


def test_wiki_search_failure_falls_back_to_local_search(monkeypatch):
    def broken(req, timeout):
        raise OSError("network down")

    monkeypatch.setattr(misinfo_infer.urllib.request, "urlopen", broken)

    assert misinfo_infer.wiki_search_titles("지구 평평") == []


def test_wiki_search_sends_user_agent_and_reads_titles(monkeypatch):
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"query": {"search": [{"title": "지평설"}, {"title": "지구"}]}}).encode()

    def fake_urlopen(req, timeout):
        seen["agent"] = req.get_header("User-agent")
        seen["url"] = req.full_url
        return FakeResponse()

    monkeypatch.setattr(misinfo_infer.urllib.request, "urlopen", fake_urlopen)

    assert misinfo_infer.wiki_search_titles("지구 평평") == ["지평설", "지구"]
    assert "VeritaeMisinfo" in seen["agent"]
    assert seen["url"].startswith("https://ko.wikipedia.org/w/api.php?")


def test_pick_diverse_limits_chunks_per_article():
    from wiki_index import Chunk

    ranked = [Chunk("선풍기 사망설", f"s{i}", i) for i in range(4)] + [Chunk("미신", "m", 9), Chunk("선풍기", "f", 10)]

    picked = misinfo_infer.pick_diverse(ranked, top_k=4)

    assert [c.title for c in picked] == ["선풍기 사망설", "선풍기 사망설", "미신", "선풍기"]


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

    misinfo_infer.ask_ollama("http://ollama", "qwen", "프롬프트", misinfo_infer.JUDGE_SCHEMA)

    assert sent["think"] is False
    assert sent["format"] == misinfo_infer.JUDGE_SCHEMA
    assert sent["options"]["num_ctx"] == misinfo_infer.OLLAMA_NUM_CTX
