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
