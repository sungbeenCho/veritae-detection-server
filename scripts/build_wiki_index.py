# scripts/build_wiki_index.py
"""kowiki-latest-pages-articles.xml.bz2 -> SQLite FTS5 인덱스.
데스크탑에서 한 번(또는 원할 때 다시) 실행하는 오프라인 배치 스크립트 - GPU를 쓰지 않는다.
사용: python build_wiki_index.py --dump <bz2 경로> --output <db 경로> --snapshot <YYYY-MM-DD> [--limit N]
"""
import argparse
import bz2
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from kiwipiepy import Kiwi

from wiki_index import add_chunk, create_index, extract_keywords, split_into_chunks, strip_wiki_markup


def iter_pages(dump_path: Path, limit: int):
    n = 0
    with bz2.open(dump_path, "rb") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            tag = elem.tag
            if not tag.endswith("}page"):
                continue
            ns_prefix = tag[: tag.index("}") + 1]
            ns = elem.findtext(f"{ns_prefix}ns")
            redirect = elem.find(f"{ns_prefix}redirect")
            if ns == "0" and redirect is None:
                title = elem.findtext(f"{ns_prefix}title")
                text = elem.findtext(f"{ns_prefix}revision/{ns_prefix}text") or ""
                yield title, text
                n += 1
                if limit and n >= limit:
                    elem.clear()
                    return
            elem.clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0=전체(디버그/테스트용 제한)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    kiwi = Kiwi()
    conn = create_index(args.output, snapshot=args.snapshot)

    pages = chunks_total = 0
    t0 = time.time()
    for title, wikitext in iter_pages(args.dump, args.limit):
        pages += 1
        plain = strip_wiki_markup(wikitext)
        for chunk in split_into_chunks(plain):
            keywords = extract_keywords(kiwi, chunk)
            if keywords:
                add_chunk(conn, title, chunk, keywords)
                chunks_total += 1
        if pages % 5000 == 0:
            conn.commit()
            elapsed = time.time() - t0
            print(f"pages={pages} chunks={chunks_total} elapsed={elapsed:.0f}s", file=sys.stderr, flush=True)

    conn.commit()
    conn.close()
    print(f"완료: pages={pages} chunks={chunks_total} elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
