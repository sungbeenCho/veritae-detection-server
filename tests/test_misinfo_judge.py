"""judge_sentence의 판정 흐름(2026-10-14): 주장 정리(LLM, 문장만) → 정리된 주장으로 위키 검색 → 판정(LLM, 정리된
주장만) → 반박이면 근거 문장 고르기(NLI 0.8 이상 중 최고점, 없으면 판정이 인용한 문단에서 이유와 가장 가까운 문장).
NLI는 판정을 버리지 않는다. Ollama, e5 임베딩, NLI 모델은 가짜로 바꾸고, 위키 인덱스는 임시 SQLite로 만든다."""
import json
import logging

import pytest
from kiwipiepy import Kiwi

import misinfo_infer
from misinfo_lib import EXTRACT_SCHEMA
from wiki_index import add_chunk, create_index, extract_keywords

SENTENCE = "선풍기를 틀고 자면 사망한다는 이야기를 들었어요."
GARBLED = "선풍기틀 틀고 자면 사망하다는 이야기틀 들없어요"
CLAIM = "선풍기를 틀고 자면 사망한다."
INTRO = "선풍기를 켜고 자면 사망한다는 가설이 있다."
REFUTING = "그러나 이 속설은 과학적 근거가 없다."
UNRELATED = "선풍기에 말려서 사망하는 장면이 나온다."
FAN_DEATH_BLOCK = f"{INTRO} {REFUTING}"


@pytest.fixture(scope="module")
def kiwi():
    return Kiwi()


@pytest.fixture
def conn(tmp_path, kiwi):
    conn = create_index(tmp_path / "wiki.sqlite3", snapshot="2026-09-01")
    for title, text in [
        ("케니 맥코믹", UNRELATED),
        ("선풍기 사망설", INTRO),
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
    """LLM 응답을 정한다. 주장 정리는 claim/checkable(문장별로 다르게 하려면 claims={문장: (claim, checkable)}),
    판정은 label과 인용할 문단 제목(cite)으로 정한다(번호는 검색 순위로 정해진다)."""
    calls = []
    answer = {}

    def fake_ask(url, model, prompt, schema):
        calls.append((schema, prompt))
        if schema == EXTRACT_SCHEMA:
            sentence = prompt.split("[문장]\n", 1)[1]
            claim, checkable = answer.get("claims", {}).get(
                sentence, (answer.get("claim", CLAIM), answer.get("checkable", True)),
            )
            return json.dumps({"claim": claim, "checkable": checkable}, ensure_ascii=False)
        ids = _block_ids(prompt)
        return json.dumps(
            {"evidence_ids": [ids[t] for t in answer["cite"] if t in ids],
             "reason": answer.get("reason", "[1] 문서는 근거가 없다고 한다."), "label": answer["label"]},
            ensure_ascii=False,
        )

    monkeypatch.setattr(misinfo_infer, "ask_ollama", fake_ask)
    monkeypatch.setattr(misinfo_infer, "rerank", lambda sentence, candidates, top_k: candidates[:top_k])
    return calls, answer


def _nli(scores: dict[str, float]):
    """주어진 문장에는 정해진 모순 확률을, 나머지는 0.01을 주는 가짜 NLI. 호출 기록도 남긴다."""
    seen = []

    def score(premises, hypothesis):
        seen.append((list(premises), hypothesis))
        return [scores.get(p, 0.01) for p in premises]

    score.seen = seen
    return score


@pytest.fixture
def embed_prefers(monkeypatch):
    """가짜 e5: 지정한 문장만 질의와 가깝게 만든다(근거 문장 고르기의 '이유와 가장 가까운 문장' 확인용)."""
    import torch

    chosen = {}

    def fake_embed(texts):
        return torch.tensor([
            [1.0, 0.0] if text.startswith("query: ") or text == f"passage: {chosen.get('text')}" else [0.0, 1.0]
            for text in texts
        ])

    monkeypatch.setattr(misinfo_infer, "embed", fake_embed)
    return chosen


def _run(kiwi, conn, nli, sentence=SENTENCE):
    return misinfo_infer.judge_sentence(kiwi, conn, sentence, "http://ollama", "qwen", 5, nli)


# --- 1단계: 주장 정리 ----------------------------------------------------------------------------------------

def test_claim_is_extracted_first_from_the_sentence_alone(kiwi, conn, llm):
    """주장 정리는 위키 문단을 보기 전에 한다 - 위키 내용에 끌려 깨진 문장을 "속설이 있다"로 정리했다(2026-10-13)."""
    calls, answer = llm
    answer.update(label="판단불가", cite=())

    _run(kiwi, conn, _nli({}))

    schema, prompt = calls[0]
    assert schema == EXTRACT_SCHEMA
    assert SENTENCE in prompt
    assert "근거 문단" not in prompt


def test_sentence_without_checkable_fact_stops_before_search_and_judgment(kiwi, conn, llm):
    """인사·의견·개인 일정 같은 문장은 위키 검색과 판정을 건너뛴다(문장당 판정 시간이 빠진다)."""
    calls, answer = llm
    answer.update(claim="", checkable=False)

    claim, timer = _judge_with_timer(kiwi, conn, _nli({}))

    assert claim is None
    assert timer.outcome == "not_checkable"
    assert len(calls) == 1
    assert list(timer.seconds) == ["extract_llm"]


def test_broken_extract_answer_is_treated_as_cannot_decide(kiwi, conn, monkeypatch, caplog):
    monkeypatch.setattr(misinfo_infer, "ask_ollama", lambda url, model, prompt, schema: "형식 깨짐")

    with caplog.at_level(logging.WARNING, logger="misinfo"):
        claim, timer = _judge_with_timer(kiwi, conn, _nli({}))

    assert claim is None
    assert timer.outcome == "extract_format_error"
    assert "search" not in timer.seconds
    assert "주장 정리 응답 형식 오류" in caplog.text


def test_search_and_judgment_use_the_extracted_claim_not_the_raw_sentence(kiwi, conn, llm):
    """깨진 OCR 문장의 오타 단어("선풍기틀")로 검색하지 않고, 판정에도 원래 문장을 보여주지 않는다."""
    calls, answer = llm
    answer.update(label="판단불가", cite=())

    _run(kiwi, conn, _nli({}), sentence=GARBLED)

    _, judge_prompt = calls[1]
    assert f"[주장]\n{CLAIM}" in judge_prompt
    assert GARBLED not in judge_prompt
    assert FAN_DEATH_BLOCK in judge_prompt  # 정리된 주장의 "선풍기"로 찾은 문서


def test_shown_sentence_is_the_original_even_though_judgment_used_the_claim(kiwi, conn, llm):
    _, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",))

    claim = _run(kiwi, conn, _nli({REFUTING: 0.99}), sentence=GARBLED)

    assert claim["sentence"] == GARBLED


# --- 2단계: 판정 ----------------------------------------------------------------------------------------------

def test_non_refutation_does_not_call_nli(kiwi, conn, llm):
    _, answer = llm
    answer.update(label="판단불가", cite=())
    nli = _nli({REFUTING: 0.99})

    assert _run(kiwi, conn, nli) is None
    assert nli.seen == []


def test_refutation_without_cited_evidence_is_not_shown(kiwi, conn, llm):
    _, answer = llm
    answer.update(label="반박", cite=())
    nli = _nli({REFUTING: 0.99})

    assert _run(kiwi, conn, nli) is None
    assert nli.seen == []


def test_llm_is_called_twice_extract_then_judge(kiwi, conn, llm):
    """LLM이 근거 문장을 고르던 단계는 없앴다(2026-10-13). 호출은 주장 정리와 판정 두 번뿐이다."""
    calls, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",))

    _run(kiwi, conn, _nli({REFUTING: 0.99}))

    assert [schema for schema, _ in calls] == [EXTRACT_SCHEMA, misinfo_infer.JUDGE_SCHEMA]


def test_block_numbers_in_reason_become_titles(kiwi, conn, llm):
    calls, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",))

    claim = _run(kiwi, conn, _nli({REFUTING: 0.99}))

    first_title = next(iter(_block_ids(calls[1][1])))
    assert claim["reason"] == f"'{first_title}' 문서는 근거가 없다고 한다."


def test_neighbor_context_is_included_in_judge_prompt(kiwi, conn, llm):
    calls, answer = llm
    answer.update(label="판단불가", cite=())

    _run(kiwi, conn, _nli({}))

    assert FAN_DEATH_BLOCK in calls[1][1]


# --- 3단계: 근거 문장 고르기 -----------------------------------------------------------------------------------

def test_nli_scores_sentences_of_all_retrieved_blocks_against_the_claim(kiwi, conn, llm):
    """판정이 엉뚱한 문단을 인용해도 맞는 근거를 놓치지 않도록, 검색된 문단 전체의 문장이 후보다."""
    _, answer = llm
    answer.update(label="반박", cite=("케니 맥코믹",))
    nli = _nli({REFUTING: 0.99})

    claim = _run(kiwi, conn, nli)

    [(premises, hypothesis)] = nli.seen
    assert set(premises) == {INTRO, REFUTING, UNRELATED}
    assert hypothesis == CLAIM
    assert claim["evidence"][0]["text"] == REFUTING


def test_only_the_highest_scoring_sentence_is_shown(kiwi, conn, llm):
    """기준(0.8)을 넘는 문장이 여러 개여도 화면에는 가장 높은 문장 하나와 문서 링크만 나간다(2026-10-01 사용자 결정)."""
    _, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",))

    claim = _run(kiwi, conn, _nli({REFUTING: 0.99, UNRELATED: 0.90}))

    assert [(e["title"], e["text"]) for e in claim["evidence"]] == [("선풍기 사망설", REFUTING)]
    assert claim["evidence"][0]["url"] == "https://ko.wikipedia.org/wiki/선풍기_사망설"
    assert claim["sentence"] == SENTENCE


def test_sentence_scoring_exactly_threshold_is_used(kiwi, conn, llm):
    _, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",))

    claim, timer = _judge_with_timer(kiwi, conn, _nli({REFUTING: 0.8}))

    assert claim["evidence"][0]["text"] == REFUTING
    assert timer.outcome == "refuted"


def test_refutation_is_kept_when_nli_finds_no_sentence(kiwi, conn, llm, embed_prefers):
    """NLI가 0.8을 넘는 문장을 못 찾아도 판정을 버리지 않는다. 위키가 "…라는 미신이다"라고만 쓰면 NLI가 0.06을 줘서
    맞는 판정이 버려졌다(2026-10-13 이미지 실측). 근거는 판정이 인용한 문단에서 이유와 가장 가까운 문장이다."""
    _, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",), reason="선풍기 사망설은 과학적 근거가 없는 미신이다.")
    embed_prefers["text"] = REFUTING

    claim, timer = _judge_with_timer(kiwi, conn, _nli({REFUTING: 0.06, UNRELATED: 0.5}))

    assert [(e["title"], e["text"]) for e in claim["evidence"]] == [("선풍기 사망설", REFUTING)]
    assert claim["reason"] == "선풍기 사망설은 과학적 근거가 없는 미신이다."
    assert timer.outcome == "refuted_reason_match"
    assert list(timer.seconds) == ["extract_llm", "search", "rerank", "judge_llm", "shortlist", "nli", "reason_match"]


def test_reason_match_only_looks_at_blocks_the_judgment_cited(kiwi, conn, llm, embed_prefers):
    """인용하지 않은 문단의 문장(UNRELATED)은 이유와 가까워도 근거로 쓰지 않는다."""
    _, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",))
    embed_prefers["text"] = UNRELATED

    claim = _run(kiwi, conn, _nli({}))

    assert claim["evidence"][0]["title"] == "선풍기 사망설"
    assert claim["evidence"][0]["text"] in {INTRO, REFUTING}


def test_closest_to_reason_picks_the_nearest_sentence(monkeypatch):
    import torch

    vectors = {"query: 이유": [1.0, 0.0], "passage: 가까움": [0.9, 0.1], "passage: 멂": [0.0, 1.0]}
    monkeypatch.setattr(misinfo_infer, "embed", lambda texts: torch.tensor([vectors[t] for t in texts]))

    assert misinfo_infer.closest_to_reason("이유", [("A", "멂"), ("B", "가까움")]) == ("B", "가까움")


def test_closest_to_reason_with_one_candidate_does_not_embed(monkeypatch):
    monkeypatch.setattr(misinfo_infer, "embed", lambda texts: (_ for _ in ()).throw(AssertionError("호출되면 안 된다")))

    assert misinfo_infer.closest_to_reason("이유", [("A", "문장")]) == ("A", "문장")


def test_shortlist_keeps_sentences_closest_to_claim(monkeypatch):
    import torch

    vectors = {"query: 주장": [1.0, 0.0], "passage: 가까움": [0.9, 0.1], "passage: 중간": [0.5, 0.5], "passage: 멂": [0.0, 1.0]}
    monkeypatch.setattr(misinfo_infer, "embed", lambda texts: torch.tensor([vectors[t] for t in texts]))
    candidates = [("A", "멂"), ("B", "중간"), ("C", "가까움")]

    assert misinfo_infer.shortlist_sentences("주장", candidates, size=2) == [("C", "가까움"), ("B", "중간")]


def test_shortlist_returns_all_when_few_candidates(monkeypatch):
    monkeypatch.setattr(misinfo_infer, "embed", lambda texts: (_ for _ in ()).throw(AssertionError("호출되면 안 된다")))

    assert misinfo_infer.shortlist_sentences("주장", [("A", "문장")], size=5) == [("A", "문장")]


def test_shortlist_default_size_is_fifteen(monkeypatch):
    """15개 기준: 2026-10-13 기록 대조(59문장)에서 5개는 24, 15개는 26개를 잡았고 20개나 전체도 26개로 같았다."""
    import torch

    assert misinfo_infer.NLI_SHORTLIST_SIZE == 15
    monkeypatch.setattr(
        misinfo_infer, "embed",
        lambda texts: torch.tensor([[1.0, 0.0]] + [[1.0 - i * 0.01, i * 0.01] for i in range(len(texts) - 1)]),
    )
    candidates = [("제목", f"문장 {i}") for i in range(20)]

    result = misinfo_infer.shortlist_sentences("주장", candidates)

    assert result == candidates[:15]


# --- 위키 검색 기능 / 검색 / Ollama 요청 ----------------------------------------------------------------------

def test_wiki_search_adds_articles_local_keyword_search_missed(kiwi, tmp_path, llm, monkeypatch):
    """로컬 키워드 검색은 단어가 다르면 못 찾는다("조작" vs "음모론") - 위키백과 검색이 찾은 문서를 후보에 더한다.
    보내는 단어는 정리된 주장에서 뽑는다."""
    conn = create_index(tmp_path / "w.sqlite3", snapshot="2026-09-01")
    for title, text in [("아폴로 11호", "아폴로 11호는 1969년 달에 착륙했다."),
                        ("달착륙 음모론", "달착륙 음모론은 아폴로 계획이 날조되었다는 주장이다. 과학자들은 이를 반박했다.")]:
        add_chunk(conn, title, text, extract_keywords(kiwi, text))
    conn.commit()
    sent = []
    monkeypatch.setattr(misinfo_infer, "wiki_search_titles", lambda terms: sent.append(terms) or ["달착륙 음모론"])
    calls, answer = llm
    answer.update(claim="아폴로 11호의 달 착륙은 조작되었다.", label="판단불가", cite=())

    misinfo_infer.judge_sentence(kiwi, conn, "아폴로 11호 달 착륙이 조작됐대요", "u", "m", 5, _nli({}), wiki_search=True)

    judge_prompt = calls[1][1]
    assert "(달착륙 음모론)" in judge_prompt
    assert sent == ["아폴로 11 달 착륙 조작"]  # 문장이 아니라 핵심 단어만 보낸다


def test_wiki_search_is_not_used_unless_enabled(kiwi, conn, llm, monkeypatch):
    monkeypatch.setattr(misinfo_infer, "wiki_search_titles", lambda terms: (_ for _ in ()).throw(AssertionError("호출되면 안 된다")))
    _, answer = llm
    answer.update(label="판단불가", cite=())

    assert _run(kiwi, conn, _nli({})) is None


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


# --- 단계별 시간과 결과(outcome) 기록(2026-10-13) -----------------------------------------------------
# 이미지 한 장이 2분 걸렸고 탐지가 비었는데, 어느 단계가 느린지 / 왜 비었는지를 로그로 알 수 없었다.
# 문장마다 단계 시간과 결과를 남긴다.

def _judge_with_timer(kiwi, conn, nli, sentence=SENTENCE):
    from misinfo_timing import StepTimer

    timer = StepTimer()
    claim = misinfo_infer.judge_sentence(kiwi, conn, sentence, "http://ollama", "qwen", 5, nli, timer=timer)
    return claim, timer


def test_timer_records_every_step_of_a_refutation(kiwi, conn, llm):
    _, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",))

    claim, timer = _judge_with_timer(kiwi, conn, _nli({REFUTING: 0.99}))

    assert claim is not None
    assert timer.outcome == "refuted"
    assert list(timer.seconds) == ["extract_llm", "search", "rerank", "judge_llm", "shortlist", "nli"]


def test_outcome_is_no_evidence_found_when_search_finds_nothing(kiwi, conn, llm):
    _, answer = llm
    answer.update(claim="!!!")

    claim, timer = _judge_with_timer(kiwi, conn, _nli({}), sentence="!!!")

    assert claim is None
    assert timer.outcome == "no_evidence_found"
    assert "judge_llm" not in timer.seconds


def test_outcome_is_not_refuted_when_llm_says_cannot_decide(kiwi, conn, llm):
    _, answer = llm
    answer.update(label="판단불가", cite=())

    claim, timer = _judge_with_timer(kiwi, conn, _nli({REFUTING: 0.99}))

    assert claim is None
    assert timer.outcome == "not_refuted"
    assert "nli" not in timer.seconds


def test_outcome_is_judge_format_error_when_judgment_answer_is_broken(kiwi, conn, monkeypatch):
    def ask(url, model, prompt, schema):
        return json.dumps({"claim": CLAIM, "checkable": True}) if schema == EXTRACT_SCHEMA else "형식 깨짐"

    monkeypatch.setattr(misinfo_infer, "ask_ollama", ask)
    monkeypatch.setattr(misinfo_infer, "rerank", lambda sentence, candidates, top_k: candidates[:top_k])

    claim, timer = _judge_with_timer(kiwi, conn, _nli({}))

    assert claim is None
    assert timer.outcome == "judge_format_error"


def test_judge_sentence_still_works_without_a_timer(kiwi, conn, llm):
    """회귀 테스트 등 기존 호출부는 timer를 넘기지 않는다."""
    _, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",))

    assert _run(kiwi, conn, _nli({REFUTING: 0.99})) is not None


def test_refutation_check_log_shows_claim_top_nli_scores_and_chosen_path(kiwi, conn, llm, embed_prefers, caplog):
    """탐지가 비었거나 근거가 이상할 때 원인을 서버 로그로 알 수 있게, 반박 문장은 정리된 주장, NLI 점수가 높은
    문장들, 근거를 어느 방식으로 골랐는지를 INFO 로그에 남긴다."""
    _, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",))
    embed_prefers["text"] = REFUTING

    with caplog.at_level(logging.INFO, logger="misinfo"):
        claim, timer = _judge_with_timer(kiwi, conn, _nli({REFUTING: 0.5, UNRELATED: 0.3}))

    assert timer.outcome == "refuted_reason_match"
    assert "refutation_check" in caplog.text
    assert f"claim={CLAIM!r}" in caplog.text
    assert "threshold=0.8" in caplog.text
    assert "(선풍기 사망설) nli=0.50" in caplog.text and REFUTING in caplog.text
    assert caplog.text.index("nli=0.50") < caplog.text.index("nli=0.30")  # 점수 높은 순
    assert "evidence_by=reason_match" in caplog.text


def test_refutation_check_is_not_logged_when_llm_did_not_say_refuted(kiwi, conn, llm, caplog):
    _, answer = llm
    answer.update(label="판단불가", cite=())

    with caplog.at_level(logging.INFO, logger="misinfo"):
        _judge_with_timer(kiwi, conn, _nli({}))

    assert "refutation_check" not in caplog.text


def test_main_logs_each_sentence_and_a_summary_without_sentence_text(kiwi, conn, llm, monkeypatch, tmp_path, capsys):
    """서버가 하위 프로세스의 stderr를 서버 로그로 올린다 - 문장 번호와 단계 시간, 결과, 전체 요약이 보여야 한다.
    사용자 문장 본문은 로그에 싣지 않는다."""
    import sys
    import types

    _, answer = llm
    answer.update(label="반박", cite=("선풍기 사망설",), claims={"!!!": ("!!!", True)})
    sentences_file = tmp_path / "sentences.json"
    sentences_file.write_text(json.dumps([SENTENCE, "!!!"], ensure_ascii=False), encoding="utf-8")
    output_file = tmp_path / "result.json"
    index_path = conn.execute("PRAGMA database_list").fetchone()[2]
    monkeypatch.setattr(misinfo_infer, "NliModel", lambda path: types.SimpleNamespace(contradiction_scores=_nli({REFUTING: 0.99})))
    monkeypatch.setattr(misinfo_infer, "unload_ollama_model", lambda url, model: None)
    monkeypatch.setattr(
        sys, "argv",
        ["misinfo_infer.py", "--sentences", str(sentences_file), "--output", str(output_file),
         "--wiki-index", index_path, "--ollama-url", "http://ollama", "--ollama-model", "qwen",
         "--nli-model", str(tmp_path / "nli_aug")],
    )
    logger = logging.getLogger("misinfo")
    handlers, level = list(logger.handlers), logger.level
    logger.handlers.clear()
    try:
        misinfo_infer.main()
    finally:
        logger.handlers[:] = handlers
        logger.setLevel(level)

    err = capsys.readouterr().err
    assert "sentence 1/2 total=" in err and "outcome=refuted" in err
    assert "sentence 2/2 total=" in err and "outcome=no_evidence_found" in err
    assert "misinfo_summary sentences=2 refuted=1 startup=" in err
    assert f"misinfo_config nli_model={tmp_path / 'nli_aug'} ollama_model=qwen wiki_search=False" in err
    assert SENTENCE not in err.replace("outcome", "")  # 본문은 싣지 않는다
    assert json.loads(output_file.read_text(encoding="utf-8"))["claims"][0]["sentence"] == SENTENCE
