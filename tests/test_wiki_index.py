import sqlite3

import pytest
from kiwipiepy import Kiwi

from scripts.wiki_index import (
    Chunk,
    add_chunk,
    create_index,
    expand_with_neighbors,
    extract_keywords,
    get_snapshot,
    search,
    split_into_chunks,
    strip_wiki_markup,
    wiki_url,
)


def test_strip_wiki_markup_removes_common_noise():
    wikitext = (
        "세종대왕은 조선의 [[제4대]] [[왕]]이다.\n"
        "[[파일:King.jpg|섬네일|세종대왕 초상화]]\n"
        "훈민정음을 창제했다.<ref>세종실록</ref>\n"
        "{| class=\"wikitable\"\n|-\n| 즉위 || 1418년\n|}\n"
    )

    plain = strip_wiki_markup(wikitext)

    assert "세종대왕은 조선의 제4대 왕이다." in plain
    assert "훈민정음을 창제했다." in plain
    assert "세종실록" not in plain
    assert "wikitable" not in plain
    assert "섬네일" not in plain


def test_split_into_chunks_groups_sentences_under_target_size():
    text = "문장 하나. 문장 둘. 문장 셋. " * 20

    chunks = split_into_chunks(text, target_chars=50)

    assert len(chunks) > 1
    assert all(len(c) <= 70 for c in chunks)  # target보다 살짝 넘는 것까진 허용(문장 단위로만 자름)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")[: len("".join(chunks).replace(" ", ""))]


def test_wiki_url_encodes_spaces_as_underscores():
    assert wiki_url("선풍기 사망설") == "https://ko.wikipedia.org/wiki/선풍기_사망설"


def test_create_index_and_search_roundtrip(tmp_path):
    db_path = tmp_path / "wiki.sqlite3"
    conn = create_index(db_path, snapshot="2026-09-01")
    add_chunk(conn, "선풍기 사망설", "선풍기 사망설이란 밀폐된 방에서 선풍기를 켜놓고 자면 사망한다는 미신이다.", "선풍기 사망설 미신")
    add_chunk(conn, "에베레스트산", "에베레스트산은 세계에서 가장 높은 산이다.", "에베레스트산 세계 가장 높다 산")
    conn.commit()

    results = search(conn, "선풍기 사망설", limit=5)

    assert results[0][0] == "선풍기 사망설"
    assert get_snapshot(conn) == "2026-09-01"


def test_search_returns_empty_list_when_nothing_matches(tmp_path):
    conn = create_index(tmp_path / "wiki.sqlite3", snapshot="2026-09-01")
    add_chunk(conn, "에베레스트산", "에베레스트산은 세계에서 가장 높은 산이다.", "에베레스트산 세계 가장 높다 산")
    conn.commit()

    assert search(conn, "복권 당첨", limit=5) == []


def test_search_handles_hyphen_in_keywords_no_crash(tmp_path):
    """Regression: FTS5 special chars like hyphen should not crash."""
    conn = create_index(tmp_path / "wiki.sqlite3", snapshot="2026-09-01")
    add_chunk(conn, "코로나-19", "코로나-19는 신종 코로나바이러스 감염증이다.", "코로나-19 바이러스")
    conn.commit()

    # Should not raise OperationalError (main requirement)
    results = search(conn, "테스트-하이픈", limit=5)
    assert results == []


def test_search_handles_plus_sign_and_finds_matches(tmp_path):
    """Regression: FTS5 special chars like + should not crash, and should find exact matches."""
    conn = create_index(tmp_path / "wiki.sqlite3", snapshot="2026-09-01")
    add_chunk(conn, "프로그래밍 언어", "C++는 고성능 프로그래밍 언어이다.", "프로그래밍 C++")
    conn.commit()

    # Should not raise OperationalError, and should find the C++ keyword when searched
    results = search(conn, "C++", limit=5)
    assert len(results) == 1
    assert results[0][0] == "프로그래밍 언어"


def test_search_handles_unterminated_quote_no_crash(tmp_path):
    """Regression: Unterminated quotes should not crash."""
    conn = create_index(tmp_path / "wiki.sqlite3", snapshot="2026-09-01")
    add_chunk(conn, "테스트", "테스트 문서입니다.", "테스트")
    conn.commit()

    # Should not raise OperationalError (main requirement)
    results = search(conn, '"unterminated', limit=5)
    assert results == []


def test_search_handles_bare_paren_no_crash(tmp_path):
    """Regression: Bare parentheses (common in disambiguation titles) should not crash."""
    conn = create_index(tmp_path / "wiki.sqlite3", snapshot="2026-09-01")
    add_chunk(conn, "제목 (동음이의)", "이것은 동음이의 문서입니다.", "제목 동음이의")
    conn.commit()

    # Should not raise OperationalError (main requirement)
    results = search(conn, "(", limit=5)
    assert results == []


def test_search_finds_chunk_when_only_some_keywords_match(tmp_path):
    """Critical 회귀(2026-09-30 리뷰): 쿼리 키워드 중 일부만 조각의 키워드와 겹쳐도 찾아야 한다.
    공백으로 이어 AND로 묶으면(예전 구현) 쿼리 키워드 전부가 한 조각에 들어있어야만 매치되는데,
    실제 문장에서 뽑은 키워드가 ~200자 조각 하나에 전부 들어있는 경우는 드물어 거의 항상 빈
    리스트를 반환했다. "틀다"가 저장된 키워드에 없어도(AND였다면 매치 실패) OR로는 찾아야 한다.
    """
    conn = create_index(tmp_path / "wiki.sqlite3", snapshot="2026-09-01")
    add_chunk(
        conn,
        "선풍기 사망설",
        "선풍기 사망설은 밀폐된 방에서 선풍기를 켜놓고 자면 산소가 부족해 사망한다는 도시전설이다.",
        "선풍기 사망설 밀폐 방 켜놓다 자다 산소 부족 사망 도시전설",
    )
    conn.commit()

    results = search(conn, "선풍기 틀다 자다 사망", limit=5)

    assert results
    assert results[0][0] == "선풍기 사망설"


def test_search_reviewer_repro_fan_death_sentence(tmp_path):
    """리뷰어가 메모리 DB로 직접 재현한 문장 1: "선풍기를 틀고 자면 사망한다."
    "켜놓고"라는 조각 안 단어와 문장의 "틀고"가 형태소가 달라 AND로는 후보를 못 찾았다.
    """
    kiwi = Kiwi()
    conn = create_index(tmp_path / "wiki.sqlite3", snapshot="2026-09-01")
    chunk_text = (
        "선풍기 사망설은 밀폐된 방에서 선풍기를 켜놓고 자면 산소가 부족해 사망할 수 있다는 "
        "도시전설로, 과학적 근거가 없다."
    )
    add_chunk(conn, "선풍기 사망설", chunk_text, extract_keywords(kiwi, chunk_text))
    conn.commit()

    keywords = extract_keywords(kiwi, "선풍기를 틀고 자면 사망한다.")
    results = search(conn, keywords, limit=5)

    assert results


def test_search_reviewer_repro_great_wall_sentence(tmp_path):
    """리뷰어가 메모리 DB로 직접 재현한 문장 2: "만리장성은 우주에서 맨눈으로 보인다."
    "육안"과 "맨눈"처럼 표현이 다른 단어가 섞이면 AND로는 후보를 못 찾았다.
    """
    kiwi = Kiwi()
    conn = create_index(tmp_path / "wiki.sqlite3", snapshot="2026-09-01")
    chunk_text = (
        "만리장성은 우주에서 육안으로 보이지 않는다는 것이 여러 우주비행사의 증언과 연구로 "
        "확인되었다."
    )
    add_chunk(conn, "만리장성", chunk_text, extract_keywords(kiwi, chunk_text))
    conn.commit()

    keywords = extract_keywords(kiwi, "만리장성은 우주에서 맨눈으로 보인다.")
    results = search(conn, keywords, limit=5)

    assert results


def _index_with(tmp_path, rows):
    conn = create_index(tmp_path / "wiki.sqlite3", snapshot="2026-09-01")
    for title, text in rows:
        add_chunk(conn, title, text, text)
    conn.commit()
    return conn


def _chunk(conn, text):
    return Chunk(*conn.execute("SELECT title, text, rowid FROM chunks WHERE text = ?", (text,)).fetchone())


def test_search_returns_rowid_for_each_chunk(tmp_path):
    conn = _index_with(tmp_path, [("A", "첫째"), ("B", "선풍기")])

    results = search(conn, "선풍기", limit=5)

    assert results == [Chunk("B", "선풍기", 2)]


def test_expand_attaches_previous_and_next_chunk_of_same_article(tmp_path):
    """선택된 조각에 "가설이 있다"만 담기고 바로 뒤 "근거가 없다"가 잘리던 문제(2026-10-01)."""
    conn = _index_with(tmp_path, [
        ("선풍기 사망설", "선풍기 사망설은 속설이다."),
        ("선풍기 사망설", "호흡장애 가설이 있다."),
        ("선풍기 사망설", "그러나 과학적 근거는 없다."),
    ])

    blocks = expand_with_neighbors(conn, [_chunk(conn, "호흡장애 가설이 있다.")])

    assert blocks == [("선풍기 사망설", "선풍기 사망설은 속설이다. 호흡장애 가설이 있다. 그러나 과학적 근거는 없다.")]


def test_expand_does_not_cross_into_other_articles(tmp_path):
    conn = _index_with(tmp_path, [("A", "a1"), ("B", "b1"), ("C", "c1")])

    blocks = expand_with_neighbors(conn, [_chunk(conn, "b1")])

    assert blocks == [("B", "b1")]


def test_expand_merges_overlapping_selections_and_keeps_rank_order(tmp_path):
    conn = _index_with(tmp_path, [
        ("A", "a1"), ("A", "a2"), ("A", "a3"), ("A", "a4"),
        ("B", "b1"), ("B", "b2"),
    ])

    blocks = expand_with_neighbors(conn, [_chunk(conn, "b2"), _chunk(conn, "a2"), _chunk(conn, "a3")])

    assert blocks == [("B", "b1 b2"), ("A", "a1 a2 a3 a4")]


def test_expand_keeps_far_apart_chunks_of_same_article_separate(tmp_path):
    conn = _index_with(tmp_path, [("A", f"a{i}") for i in range(1, 8)])

    blocks = expand_with_neighbors(conn, [_chunk(conn, "a2"), _chunk(conn, "a6")])

    assert blocks == [("A", "a1 a2 a3"), ("A", "a5 a6 a7")]


def test_expand_returns_empty_for_no_chunks(tmp_path):
    conn = _index_with(tmp_path, [("A", "a1")])

    assert expand_with_neighbors(conn, []) == []
