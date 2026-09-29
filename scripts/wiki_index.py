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


def extract_keywords(kiwi: Kiwi, text: str) -> str:
    # 조사/어미를 떼고 의미 있는 형태소(명사/동사/형용사)만 남겨 검색 정확도를 높인다
    # ("만리장성은"으로 검색해도 "만리장성" 문서를 찾도록 - 2026-09-29 실측으로 확인된 필요성).
    keep_tags = {"NNG", "NNP", "VV", "VA", "SL", "SN"}
    tokens = [t.form for t in kiwi.tokenize(text) if t.tag in keep_tags]
    return " ".join(tokens)


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


def search(conn: sqlite3.Connection, keywords: str, limit: int = 50) -> list[tuple[str, str]]:
    if not keywords.strip():
        return []
    quoted = " ".join(_quote_fts5_token(t) for t in keywords.split())
    rows = conn.execute(
        "SELECT title, text FROM chunks WHERE keywords MATCH ? ORDER BY rank LIMIT ?",
        (quoted, limit),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]
