"""위키 검색 단계만 돌려, 주제 문서가 어디서 빠지는지 기록하는 측정 도구. 데스크탑에서 한 번 돌린다(LLM/NLI 불필요).

배경(2026-10-16): 판정에 들어가는 문단 5개에 주제와 상관없는 문서(소설·드라마·인물 문서)가 거의 항상 섞이고, 주제 문서가
아예 빠지는 문장도 있다(10-01 기록 사실 문장 53개 중 10개). 그런데 지금 로그에는 키워드 검색 50개 순위와 e5 점수가 남지
않아, 주제 문서가 (가) 키워드 검색 50개에는 들어왔는데 e5 재정렬·문서당 2개 규칙에서 밀렸는지, (나) 50개에 아예 없었는지,
(다) 그 이름의 문서 자체가 색인에 없는지(넘겨주기 문서는 색인에 안 들어감) 가릴 수 없다. 검색 구조를 바꾸기 전에 이걸 숫자로 본다.

서버와 같은 방식(키워드 추출 -> 키워드 검색 50개 -> e5 재정렬 -> 문서당 2개로 5개)을 그대로 쓰고, 중간 결과를 전부 남긴다.
"주제 문서 후보"는 문장의 명사(앞뒤로 붙은 명사는 이어 붙인 것도)와 제목이 똑같은 문서로 본다 - 정답 표시가 아니라
"이 이름의 문서가 색인에 있는가"를 보는 기준이다.

사용(text-extraction 환경, C:\\ai\\veritae-detection-server\\scripts):
  python measure_retrieval.py --wiki-index ..\\data\\wiki_index.sqlite3 --dump ..\\retrieval_dump.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import misinfo_infer
from kiwipiepy import Kiwi
from wiki_index import extract_keywords, lead_rowid, search

CASE_FILES = [
    "cases_basic.json", "cases_e2e.json", "cases_grounding.json", "cases_hard.json",
    "cases_heldout.json", "cases_realistic.json",
]
CASES_DIR = Path(__file__).resolve().parent.parent / "tests" / "regression"
NOUN_TAGS = {"NNG", "NNP"}


def subject_titles(kiwi: Kiwi, sentence: str) -> list[str]:
    """문장의 명사와, 바로 붙어 나온 명사들을 이어 붙인 말(세종+대왕 -> 세종대왕)을 중복 없이 순서대로."""
    tokens = kiwi.tokenize(sentence)
    titles: list[str] = []
    run: list = []

    def flush() -> None:
        for size in range(len(run), 0, -1):
            for start in range(len(run) - size + 1):
                part = run[start:start + size]
                # 원문에서 띄어 쓰지 않고 붙어 있던 명사만 이어 붙인다.
                if all(part[k].start + part[k].len == part[k + 1].start for k in range(len(part) - 1)):
                    word = "".join(t.form for t in part)
                    if word not in titles:
                        titles.append(word)

    for token in tokens:
        if token.tag in NOUN_TAGS:
            run.append(token)
        else:
            flush()
            run = []
    flush()
    return titles


def measure(kiwi: Kiwi, conn, sentence: str, evidence_count: int) -> dict:
    keywords = extract_keywords(kiwi, sentence)
    chunks = search(conn, keywords, limit=50) if keywords else []
    record = {"keywords": keywords, "fts": [], "final": [], "subjects": []}
    if chunks:
        query_vec = misinfo_infer.embed(["query: " + sentence])
        passage_vecs = misinfo_infer.embed([f"passage: {c.title} {c.text}" for c in chunks])
        scores = (passage_vecs @ query_vec.T).squeeze(-1).tolist()
        record["fts"] = [
            {"fts_rank": i + 1, "title": c.title, "rowid": c.rowid, "e5": round(s, 4), "text": c.text[:120]}
            for i, (c, s) in enumerate(zip(chunks, scores))
        ]
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])
        final = misinfo_infer.pick_diverse([chunks[i] for i in order], evidence_count)
        record["final"] = [c.title for c in final]

    fts_titles = [item["title"] for item in record["fts"]]
    for title in subject_titles(kiwi, sentence):
        exists = lead_rowid(conn, title) is not None
        if not exists:
            where = "no_doc"  # 이 이름의 문서가 색인에 없음(넘겨주기 별칭이거나 문서가 없음)
        elif title in record["final"]:
            where = "final"  # 판정 입력 5개에 들어감
        elif title in fts_titles:
            where = "dropped_by_rerank"  # 키워드 검색 50개에는 있었는데 재정렬·문서당 2개에서 밀림
        else:
            where = "not_in_fts"  # 문서는 있는데 키워드 검색 50개에 없음
        entry = {"title": title, "where": where}
        if where in ("final", "dropped_by_rerank"):
            hits = [item for item in record["fts"] if item["title"] == title]
            entry["best_fts_rank"] = min(item["fts_rank"] for item in hits)
            entry["best_e5"] = max(item["e5"] for item in hits)
        record["subjects"].append(entry)
    return record


def load_cases(paths: list[Path]) -> list[dict]:
    cases = []
    for path in paths:
        for case in json.loads(path.read_text(encoding="utf-8")):
            cases.append({**case, "file": path.name})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-index", required=True, type=Path)
    parser.add_argument("--dump", required=True, type=Path, help="문장별 전체 기록을 저장할 JSON 경로")
    parser.add_argument("--cases", nargs="*", type=Path, help="사례 파일(기본: tests/regression의 사례 파일 전부)")
    parser.add_argument("--evidence-count", type=int, default=5)
    args = parser.parse_args()

    paths = args.cases or [CASES_DIR / name for name in CASE_FILES if (CASES_DIR / name).exists()]
    cases = load_cases(paths)
    conn = sqlite3.connect(f"file:{args.wiki_index.resolve().as_posix()}?mode=ro", uri=True)
    kiwi = Kiwi()

    records = []
    start = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        record = measure(kiwi, conn, case["sentence"], args.evidence_count)
        records.append({**case, **record})
        # 결과 파일을 옮기지 못해도 화면 출력만 복사해 보내면 분석할 수 있게 문장마다 한 줄로 남긴다.
        marks = {"final": "들어감", "dropped_by_rerank": "밀림", "not_in_fts": "검색안됨", "no_doc": "문서없음"}
        subjects = " ".join(
            f"{s['title']}={marks[s['where']]}"
            + (f"(검색{s['best_fts_rank']}위,e5={s['best_e5']:.3f})" if "best_fts_rank" in s else "")
            for s in record["subjects"]
        )
        print(
            f"{index}/{len(cases)} [{case.get('expected', '')}] {case['sentence']} || 최종: {', '.join(record['final'])}"
            f" || 주제: {subjects}",
            flush=True,
        )
    seconds = time.perf_counter() - start

    args.dump.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    counts: dict[str, int] = {}
    for record in records:
        for subject in record["subjects"]:
            counts[subject["where"]] = counts.get(subject["where"], 0) + 1
    print()
    print(f"문장 {len(records)}개, {seconds:.0f}초 (문장당 {seconds / max(len(records), 1):.2f}초)")
    print("문장 속 명사와 제목이 같은 문서가 어디까지 왔나(명사 기준 건수):")
    for key, label in [
        ("final", "판정 입력 5개에 들어감"),
        ("dropped_by_rerank", "검색 50개엔 있었는데 재정렬에서 밀림"),
        ("not_in_fts", "문서는 있는데 검색 50개에 없음"),
        ("no_doc", "그 이름의 문서가 색인에 없음"),
    ]:
        print(f"  {label}: {counts.get(key, 0)}")
    print(f"전체 기록: {args.dump}")


if __name__ == "__main__":
    main()
