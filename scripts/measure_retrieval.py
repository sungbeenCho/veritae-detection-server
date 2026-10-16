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
from misinfo_lib import fit_blocks
from wiki_index import _quote_fts5_token, expand_with_neighbors, extract_keywords, lead_rowid, search, title_chunks

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
        # 판정 AI가 실제로 읽는 입력(앞뒤 조각·첫 문단을 붙이고 글자 수 상한을 적용한 것)의 길이.
        record["judge_input_chars"] = sum(len(t) + len(b) for t, b in fit_blocks(expand_with_neighbors(conn, final)))

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


def gold_titles(gold: dict) -> list[str]:
    return gold["gold_docs"] + gold.get("allowed_docs", [])


def gold_status(record: dict, gold: dict) -> str:
    """정답표 기준 이 문장의 결과 한 줄: 정답/허용 문서가 최종 5개·검색 50개 어디까지 왔는지."""
    fts_best: dict[str, int] = {}
    for item in record["fts"]:
        fts_best.setdefault(item["title"], item["fts_rank"])
    parts = []
    for kind, titles in (("정답", gold["gold_docs"]), ("허용", gold.get("allowed_docs", []))):
        for title in titles:
            if title in record["final"]:
                where = "최종"
            elif title in fts_best:
                where = f"밀림(검색{fts_best[title]}위)"
            elif title in record.get("no_doc", []):
                where = "색인에문서없음"
            elif title in record.get("deep_rank", {}):
                where = f"검색{record['deep_rank'][title]}위"
            else:
                where = "검색2000위밖"
            parts.append(f"{kind}:{title}={where}")
    other = [t for t in dict.fromkeys(record["final"]) if t not in gold_titles(gold)]
    return f"{' '.join(parts)} | 그 밖의 문서: {', '.join(other) or '-'}"


def deep_rank(conn, keywords: str, titles: list[str], limit: int = 2000) -> dict[str, int]:
    """검색 50개 밖으로 밀린 정답 문서가 키워드 검색으로 몇 위인지(그 문서 조각 중 가장 높은 순위). e5는 쓰지 않는다."""
    found: dict[str, int] = {}
    for rank, chunk in enumerate(search(conn, keywords, limit=limit), start=1):
        if chunk.title in titles and chunk.title not in found:
            found[chunk.title] = rank
    return found


def keyword_presence(conn, title: str, keywords: list[str]) -> str:
    """검색어 하나하나가 그 문서 색인에 몇 조각 들어 있는지. 0이면 형태소가 다르게 저장돼 검색어와 안 맞는 것이다."""
    counts = []
    for word in keywords:
        try:
            row = conn.execute(
                "SELECT count(*) FROM chunks WHERE chunks MATCH ? AND title = ?",
                (f"title : {_quote_fts5_token(title)} AND keywords : {_quote_fts5_token(word)}", title),
            ).fetchone()
            counts.append(f"{word}:{row[0]}")
        except sqlite3.OperationalError:
            counts.append(f"{word}:?")
    return " ".join(counts)


def describe_doc(conn, title: str, keywords: list[str], chunk_count: int) -> list[str]:
    """정답 문서가 색인에 어떻게 들어 있는지: 존재 여부, 검색어별 조각 수, 첫 조각과 검색어가 가장 많이 맞는 조각의 본문·저장된 색인 단어."""
    if lead_rowid(conn, title) is None:
        return [f"  [{title}] 색인에 이 제목의 문서 없음"]
    lines = [f"  [{title}] 검색어별 조각 수: {keyword_presence(conn, title, keywords)}"]
    for i, chunk in enumerate(title_chunks(conn, title, " ".join(keywords), limit=chunk_count)):
        stored = conn.execute("SELECT keywords FROM chunks WHERE rowid = ?", (chunk.rowid,)).fetchone()[0]
        # 첫 조각(정의 문단)은 앞부분만, 검색어가 맞는 조각은 근거 문장을 확인할 수 있게 자르지 않는다.
        lines.append(f"    {'첫 조각' if i == 0 else '맞는 조각'}: {chunk.text[:120] if i == 0 else chunk.text}")
        lines.append(f"      색인 단어: {stored[:100]}")
    return lines


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
    parser.add_argument("--gold", type=Path, help="정답표(tests/regression/retrieval_gold.json). 주면 정답 문서 기준으로 채점한다")
    parser.add_argument("--evidence-count", type=int, default=5)
    args = parser.parse_args()

    paths = args.cases or [CASES_DIR / name for name in CASE_FILES if (CASES_DIR / name).exists()]
    cases = load_cases(paths)
    gold = {g["sentence"]: g for g in json.loads(args.gold.read_text(encoding="utf-8"))} if args.gold else {}
    conn = sqlite3.connect(f"file:{args.wiki_index.resolve().as_posix()}?mode=ro", uri=True)
    kiwi = Kiwi()

    # 결과 파일을 옮기지 못해도 화면 출력만 복사해 보내면 분석할 수 있게 한다. 출력이 길어서 구획으로 나눠
    # 여러 번에 붙여 넣을 수 있게 하고, 정답 문서를 찾은 문장은 한 줄로 짧게, 놓친 문장만 자세히 쓴다.
    print("===== [1/3] 문장별 결과 =====")
    records = []
    start = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        record = measure(kiwi, conn, case["sentence"], args.evidence_count)
        g = gold.get(case["sentence"])
        head = f"{index}/{len(cases)} [{case.get('expected', '')}] {case['sentence']} || 입력 {record.get('judge_input_chars', 0)}자"
        if g and g["gold_docs"]:
            record["hit"] = any(t in record["final"] for t in gold_titles(g))
            if not record["hit"]:
                fts_titles = {item["title"] for item in record["fts"]}
                missing = [t for t in gold_titles(g) if t not in fts_titles]
                record["no_doc"] = [t for t in missing if lead_rowid(conn, t) is None]
                searchable = [t for t in missing if t not in record["no_doc"]]
                record["deep_rank"] = deep_rank(conn, record["keywords"], searchable) if searchable else {}
            line = f"{head} || 적중" if record["hit"] else f"{head} || 놓침 | 검색어: {record['keywords']} | {gold_status(record, g)}"
        else:
            line = f"{head} || 최종: {', '.join(record['final'])}"
        print(line, flush=True)
        records.append({**case, **record})
    seconds = time.perf_counter() - start

    args.dump.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print(f"문장 {len(records)}개, {seconds:.0f}초 (문장당 {seconds / max(len(records), 1):.2f}초)")
    if gold:
        print_gold_summary(conn, records, gold)
    print(f"전체 기록: {args.dump}")


def print_gold_summary(conn, records: list[dict], gold: dict) -> None:
    """정답표 기준 요약: 같은 문장은 한 번만 센다. 묶음별 적중 수, 정답표에 없는데 들어온 문서(사람이 판정할 목록), 정답 문서 정보."""
    first: dict[str, dict] = {}
    for record in records:
        first.setdefault(record["sentence"], record)
    items = [
        {"g": gold[s], "hit": r["hit"], "record": r}
        for s, r in first.items() if s in gold and gold[s]["gold_docs"]
    ]

    def line(label: str, group: list[dict]) -> None:
        print(f"  {label}: {sum(1 for i in group if i['hit'])}/{len(group)}")

    print()
    print("===== [2/3] 요약 =====")
    print("정답 문서 적중(같은 문장은 한 번만):")
    for status in ("확정", "약함(별도 집계)"):
        group = [i for i in items if i["g"]["gold_status"] == status]
        line(f"정답 {status}", group)
        for expected, label in (("contradiction", "거짓 문장"), ("entailment", "지지 문장")):
            line(f"  - {label}", [i for i in group if i["g"]["expected"] == expected])
    confirmed = [i for i in items if i["g"]["gold_status"] == "확정"]
    for tag in ("전언·구어", "부정", "OCR", "소개"):
        line(f"태그 {tag} (정답 확정만)", [i for i in confirmed if tag in i["g"]["tags"]])
    topics: dict[str, list[dict]] = {}
    for item in confirmed:
        topics.setdefault(item["g"]["topic"], []).append(item)
    rate = sum(sum(i["hit"] for i in v) / len(v) for v in topics.values()) / max(len(topics), 1)
    print(f"  주제 단위 평균 적중률(정답 확정 주제만): {rate:.2f} (주제 {len(topics)}개)")
    no_gold = [r for s, r in first.items() if not (gold.get(s) or {}).get("gold_docs")]
    if no_gold:
        avg = sum(r.get("judge_input_chars", 0) for r in no_gold) / len(no_gold)
        print(f"정답 없음·일상 문장 {len(no_gold)}개(중복 제외): 판정 입력 평균 {avg:.0f}자")

    print("정답표에 없는데 최종 5개에 들어온 문서(관련 있음/없음을 사람이 판정할 목록, 들어온 횟수):")
    others: dict[str, int] = {}
    for item in items:
        for title in dict.fromkeys(item["record"]["final"]):
            if title not in gold_titles(item["g"]):
                others[title] = others.get(title, 0) + 1
    print("  " + ", ".join(f"{t}({n})" for t, n in sorted(others.items(), key=lambda kv: -kv[1])))

    print()
    print("===== [3/3] 정답 문서 정보 =====")
    # 문서마다 한 번만 찍는다. 검색어는 그 문서를 정답으로 둔 모든 문장의 검색어를 모은 것이다. 놓친 문장의 정답 문서와
    # "위키에 반박 내용이 있는지" 확인이 필요한 문서(번개·금붕어)는 맞는 조각을 3개까지 찍는다.
    doc_keywords: dict[str, list[str]] = {}
    deep: set[str] = {"번개", "카타툼보 번개", "금붕어"}
    for item in items:
        for title in gold_titles(item["g"]):
            words = doc_keywords.setdefault(title, [])
            for word in item["record"]["keywords"].split():
                if word not in words:
                    words.append(word)
            if not item["hit"]:
                deep.add(title)
    for title, words in doc_keywords.items():
        for text in describe_doc(conn, title, words, 3 if title in deep else 1):
            print(text)


if __name__ == "__main__":
    main()
