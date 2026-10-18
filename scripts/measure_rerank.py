"""검색 순서 매기기만 비교하는 측정 도구(2026-10-18). 4B는 돌리지 않는다.

배경: 같은 내용인데 표현만 다른 두 문장("고래가 물고기라고 들었어요" / "고래는 물고기다")으로 검색하면 판정에 넘기는 상위 5개
문서가 통째로 달라졌다 - e5가 문장과 문단을 따로 숫자로 바꿔 비교하기 때문에 "고래 이야기가 많은 문단"이 다 비슷한 점수를 받는다.
리랭커(BAAI/bge-reranker-v2-m3, Apache-2.0, 다국어)는 문장과 문단을 한 쌍으로 같이 읽고 "이 문단이 이 문장에 답이 되는가"를
점수로 매긴다. 다음을 비교한다:
  - e5: 지금 서버(단어 검색 50개 -> e5 순서 -> 문서당 2개까지 상위 5개)
  - 제목검색: e5 + 문장 속 명사와 같은 제목의 문서를 후보에 넣고 최대 2개에 상위 자리(measure_final.py와 같음)
  - 리랭커: 단어 검색 50개 -> 리랭커 순서 -> 문서당 2개까지 상위 5개
  - 리랭커+제목: 제목이 같은 문서도 후보에 넣고 리랭커 순서(자리 보장 없음)
입력은 정답 문서표(tests/regression/retrieval_gold.json, 확정 103문장)의 원래 문장과, 같은 문장에서 4B가 뽑은 내용(어제 측정 기록
speaker_pipeline_dump.json의 분류기v2 claim) 두 가지다.

재는 것: 맞는 문서가 상위 5개 안에 든 비율, 1등이 맞는 문서인 비율, 원래 문장과 뽑은 내용의 1등이 같은 비율, 문장당 시간.
판단 기준(측정 전에 정함): 리랭커(또는 리랭커+제목)가 e5보다 상위 5개 적중이 높고, 뽑은 내용으로 검색해도 원래 문장만큼 적중하고,
문장당 시간이 서버에서 감당할 수준(문장당 리랭커 추가 3초 이하)이면 다음 단계(판정까지 붙여 측정). 아니면 멈춘다.

사용(text-extraction 환경, scripts 폴더). 처음 한 번은 리랭커 모델(약 2GB)을 내려받는다:
  python measure_rerank.py --wiki-index ..\\data\\wiki_index.sqlite3 --dump ..\\speaker_pipeline_dump.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import misinfo_infer
from kiwipiepy import Kiwi
from measure_final import title_aware_rerank
from measure_retrieval import CASES_DIR
from wiki_index import extract_keywords, search

RERANKER_ID = "BAAI/bge-reranker-v2-m3"
METHODS = ("e5", "제목검색", "리랭커", "리랭커+제목")


class Reranker:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.to(self.device)
        self.load_seconds = time.perf_counter() - start

    def order(self, text: str, candidates: list) -> list:
        scores = []
        for i in range(0, len(candidates), 16):
            batch = candidates[i:i + 16]
            inputs = self.tokenizer([text] * len(batch), [f"{c.title} {c.text}" for c in batch], padding=True,
                                    truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                scores += self.model(**inputs).logits.view(-1).float().cpu().tolist()
        return [c for _, c in sorted(zip(scores, range(len(candidates))), key=lambda x: -x[0])]


def top5(order_fn, text, candidates):
    ranked = order_fn(text, candidates)
    return misinfo_infer.pick_diverse(ranked, 5)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-index", required=True, type=Path)
    parser.add_argument("--dump", required=True, type=Path, help="measure_speaker_pipeline.py 기록(뽑은 내용을 꺼낸다)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    gold = [g for g in json.loads((CASES_DIR / "retrieval_gold.json").read_text(encoding="utf-8"))
            if g.get("gold_status") == "확정" and g.get("gold_docs")]
    claims = {r["sentence"]: r["분류기v2"].get("claim") for r in json.loads(args.dump.read_text(encoding="utf-8"))}
    conn = sqlite3.connect(f"file:{args.wiki_index.resolve().as_posix()}?mode=ro", uri=True)
    kiwi = Kiwi()
    title_rerank = title_aware_rerank(kiwi, conn)
    misinfo_infer.embed(["query: 시험"])  # e5를 미리 올린다
    reranker = Reranker(RERANKER_ID)
    print(f"리랭커 장치: {reranker.device}, 로드 {reranker.load_seconds:.1f}초 / 정답 문서표 문장 {len(gold)}개")

    def e5_order(text, cands):
        query = misinfo_infer.embed(["query: " + text])
        passages = misinfo_infer.embed([f"passage: {c.title} {c.text}" for c in cands])
        return [cands[i] for i in (passages @ query.T).squeeze(-1).argsort(descending=True).tolist()]

    def with_titles(text, cands):
        from measure_retrieval import subject_titles
        from wiki_index import title_chunks
        keywords = extract_keywords(kiwi, text)
        seen, pool = {c.rowid for c in cands}, list(cands)
        for t in [t for t in subject_titles(kiwi, text) if len(t) >= 2][:6]:
            for c in title_chunks(conn, t, keywords):
                if c.rowid not in seen:
                    seen.add(c.rowid)
                    pool.append(c)
        return pool

    runners = {
        "e5": lambda text, cands: top5(e5_order, text, cands),
        "제목검색": lambda text, cands: title_rerank(text, cands, 5),
        "리랭커": lambda text, cands: top5(reranker.order, text, cands),
        "리랭커+제목": lambda text, cands: top5(reranker.order, text, with_titles(text, cands)),
    }
    rows, seconds = [], {m: 0.0 for m in METHODS}
    start = time.perf_counter()
    for index, g in enumerate(gold, start=1):
        docs = set(g["gold_docs"]) | set(g.get("allowed_docs") or [])
        row = {"sentence": g["sentence"], "claim": claims.get(g["sentence"]), "gold": sorted(docs)}
        for kind, text in (("원문", g["sentence"]), ("정리", claims.get(g["sentence"]))):
            if not text:
                continue
            cands = search(conn, extract_keywords(kiwi, text), limit=50)
            for m in METHODS:
                t0 = time.perf_counter()
                picked = runners[m](text, cands) if cands else []
                seconds[m] += time.perf_counter() - t0
                titles = [c.title for c in picked]
                row[f"{m}/{kind}"] = {"titles": titles, "hit5": bool(docs & set(titles)), "hit1": bool(titles) and titles[0] in docs}
        rows.append(row)
        if index % 10 == 0 or index == len(gold):
            elapsed = time.perf_counter() - start
            print(f"진행 {index}/{len(gold)}, {elapsed:.0f}초 지남, 남은 시간 약 {elapsed / index * (len(gold) - index):.0f}초", flush=True)

    print("\n===== 요약 =====")
    for m in METHODS:
        for kind in ("원문", "정리"):
            got = [r[f"{m}/{kind}"] for r in rows if f"{m}/{kind}" in r]
            print(f"{m} / {kind}: 상위5 적중 {sum(x['hit5'] for x in got)}/{len(got)}, 1등 적중 {sum(x['hit1'] for x in got)}/{len(got)}")
        both = [r for r in rows if f"{m}/원문" in r and f"{m}/정리" in r]
        same1 = sum(1 for r in both if r[f"{m}/원문"]["titles"][:1] == r[f"{m}/정리"]["titles"][:1])
        print(f"  {m}: 원문과 정리 내용의 1등 문서가 같음 {same1}/{len(both)}, 문장당 시간(두 입력 합) {seconds[m] / max(len(rows), 1):.2f}초")
    print("\n===== 리랭커가 e5와 다른 결과(정리 내용 기준, 상위5 적중이 갈린 문장) =====")
    for r in rows:
        a, b = r.get("e5/정리"), r.get("리랭커/정리")
        if a and b and a["hit5"] != b["hit5"]:
            print(f"  [{'리랭커만 맞힘' if b['hit5'] else 'e5만 맞힘'}] {r['sentence']} (정답 {', '.join(r['gold'])})")
            print(f"      e5: {', '.join(a['titles'])}")
            print(f"      리랭커: {', '.join(b['titles'])}")
    if args.out:
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
