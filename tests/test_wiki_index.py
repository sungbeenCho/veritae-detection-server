import sqlite3

import pytest

from scripts.wiki_index import (
    add_chunk,
    create_index,
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
