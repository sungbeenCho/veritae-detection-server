"""위키 조각 인덱스 공용 모듈. build_wiki_index.py(구축)와 misinfo_infer.py(조회) 둘 다
이 모듈을 쓴다 - 같은 디렉터리 안 스크립트라 파이썬이 자동으로 sys.path에 잡아준다.
RAM에 통째로 올리지 않고 디스크의 SQLite FTS5 파일 하나로 검색한다
(docs/superpowers/specs/2026-09-29-misinformation-detection-design.md §4).
"""
from __future__ import annotations

import re
import sqlite3
import urllib.parse
from pathlib import Path
from typing import NamedTuple

import mwparserfromhell
from kiwipiepy import Kiwi

_DROP_TAGS = {"ref", "gallery", "math", "timeline", "score"}
_DROP_LINE = re.compile(r"^\s*(분류:|파일:|File:|Category:|thumb\||섬네일\|)", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def strip_wiki_markup(wikitext: str) -> str:
    code = mwparserfromhell.parse(wikitext)
    for tag in code.filter_tags(recursive=True):
        if str(tag.tag).lower() in _DROP_TAGS:
            try:
                code.remove(tag)
            except ValueError:
                pass
    plain = code.strip_code(normalize=True, collapse=True)
    lines = [line for line in plain.split("\n") if line.strip() and not _DROP_LINE.match(line.strip())]
    return "\n".join(lines)


def split_into_chunks(text: str, target_chars: int = 200) -> list[str]:
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        sentences.extend(s.strip() for s in _SENT_SPLIT.split(line) if s.strip())

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > target_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


_KEEP_TAGS = {"NNG", "NNP", "VV", "VA", "SL", "SN"}


def extract_keywords(kiwi: Kiwi, text: str) -> str:
    # 조사/어미를 떼고 의미 있는 형태소(명사/동사/형용사)만 남겨 검색 정확도를 높인다
    # ("만리장성은"으로 검색해도 "만리장성" 문서를 찾도록 - 2026-09-29 실측으로 확인된 필요성).
    tokens = [t.form for t in kiwi.tokenize(text) if t.tag in _KEEP_TAGS]
    return " ".join(tokens)


def extract_keywords_batch(kiwi: Kiwi, texts: list[str]) -> list[str]:
    """extract_keywords와 동일한 결과를 텍스트 여러 개를 한 번에 넘겨 얻는다.

    kiwipiepy는 문자열 리스트를 한 번에 넘기면 내부 워커 스레드로 병렬 처리한다 -
    build_wiki_index.py처럼 청크가 수백만 개인 일괄 처리에서만 쓴다. 실시간 판정
    한 문장 처리(misinfo_infer.py)는 병렬화할 게 없으니 extract_keywords를 그대로 쓴다.
    """
    return [
        " ".join(t.form for t in tokens if t.tag in _KEEP_TAGS)
        for tokens in kiwi.tokenize(texts)
    ]


def wiki_url(title: str) -> str:
    return f"https://ko.wikipedia.org/wiki/{title.replace(' ', '_')}"


def create_index(db_path: Path, snapshot: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(title, text, keywords)"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('snapshot', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (snapshot,),
    )
    conn.commit()
    return conn


def add_chunk(conn: sqlite3.Connection, title: str, text: str, keywords: str) -> None:
    conn.execute("INSERT INTO chunks (title, text, keywords) VALUES (?, ?, ?)", (title, text, keywords))


def get_snapshot(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT value FROM meta WHERE key = 'snapshot'").fetchone()
    return row[0] if row else ""


def _quote_fts5_token(token: str) -> str:
    """Escape FTS5 special characters by quoting tokens as literal phrases."""
    return '"' + token.replace('"', '""') + '"'


class Chunk(NamedTuple):
    title: str
    text: str
    rowid: int


def search(conn: sqlite3.Connection, keywords: str, limit: int = 50) -> list[Chunk]:
    if not keywords.strip():
        return []
    # OR로 이어야 한다 - AND(공백 join)는 문장에서 뽑은 키워드가 ~200자 조각 하나에 전부
    # 들어있어야만 매치되는데 그런 경우가 드물어 거의 항상 빈 리스트를 반환한다
    # (2026-09-30 리뷰에서 실제 재현: "선풍기를 틀고 자면 사망한다." 등). OR + BM25
    # 랭킹(ORDER BY rank)으로 일부만 일치해도 후보를 찾고 관련도 순으로 정렬한다.
    quoted = " OR ".join(_quote_fts5_token(t) for t in keywords.split())
    rows = conn.execute(
        "SELECT title, text, rowid FROM chunks WHERE keywords MATCH ? ORDER BY rank LIMIT ?",
        (quoted, limit),
    ).fetchall()
    return [Chunk(*row) for row in rows]


def expand_with_neighbors(conn: sqlite3.Connection, chunks: list[Chunk]) -> list[tuple[str, str]]:
    """각 조각에 같은 문서의 바로 앞뒤 조각을 붙여 (제목, 본문) 묶음으로 돌려준다.

    약 200자 조각 하나만 보면 "이런 가설이 있다"까지만 담기고 바로 뒤의 "과학적 근거가
    없다"가 잘려, 판정도 근거 표시도 틀어진다(2026-10-01 선풍기 사망설 실측). 구축 시 한
    문서의 조각은 연속된 rowid로 저장되므로 인덱스를 다시 만들지 않고 rowid±1로 찾는다.
    겹치거나 맞닿는 묶음은 하나로 합치고, 묶음 순서는 그 안에 든 선택 조각의 순위를 따른다.
    """
    if not chunks:
        return []
    wanted: dict[int, set[str]] = {}
    for chunk in chunks:
        for rowid in (chunk.rowid - 1, chunk.rowid, chunk.rowid + 1):
            wanted.setdefault(rowid, set()).add(chunk.title)
    placeholders = ",".join("?" * len(wanted))
    rows = conn.execute(
        f"SELECT rowid, title, text FROM chunks WHERE rowid IN ({placeholders})", list(wanted)
    ).fetchall()
    kept = {rowid: (title, text) for rowid, title, text in rows if title in wanted[rowid]}

    rank = {chunk.rowid: i for i, chunk in enumerate(chunks)}
    runs: list[list[int]] = []
    for rowid in sorted(kept):
        if runs and runs[-1][-1] == rowid - 1 and kept[runs[-1][-1]][0] == kept[rowid][0]:
            runs[-1].append(rowid)
        else:
            runs.append([rowid])
    runs.sort(key=lambda run: min(rank.get(r, len(chunks)) for r in run))
    return [(kept[run[0]][0], " ".join(kept[r][1] for r in run)) for run in runs]
