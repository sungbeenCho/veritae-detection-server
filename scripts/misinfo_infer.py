# scripts/misinfo_infer.py
"""문장 목록 -> 위키 검색 -> e5 재정렬 -> LLM 판정 -> 반박된 문장만 결과로.
text-extraction conda env(kiwipiepy, mwparserfromhell, torch, transformers 설치됨)에서
실행되어야 한다. veritae-detection-server(FastAPI)는 이 스크립트를 subprocess로 호출하고
--output 경로의 JSON만 읽는다 - scam_infer.py와 동일한 패턴.

docs(veritae-server 레포): docs/superpowers/specs/2026-09-29-misinformation-detection-design.md
"""
import argparse
import json
import urllib.request
from pathlib import Path

from kiwipiepy import Kiwi

from misinfo_lib import build_claim, build_prompt, parse_llm_response
from wiki_index import extract_keywords, search

OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["지지", "반박", "판단불가"]},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
}


def embed(texts: list[str]):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
    model.eval()
    batch = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    vectors = (hidden * mask).sum(1) / mask.sum(1)
    return torch.nn.functional.normalize(vectors, dim=-1)


def rerank(sentence: str, candidates: list[tuple[str, str]], top_k: int) -> list[tuple[str, str]]:
    if not candidates:
        return []
    query_vec = embed(["query: " + sentence])
    passage_vecs = embed([f"passage: {title} {text}" for title, text in candidates])
    scores = (passage_vecs @ query_vec.T).squeeze(-1)
    top_indices = scores.argsort(descending=True)[:top_k].tolist()
    return [candidates[i] for i in top_indices]


def ask_ollama(ollama_url: str, model: str, prompt: str) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "format": OLLAMA_SCHEMA,
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["response"]


def judge_sentence(kiwi: Kiwi, conn, sentence: str, ollama_url: str, model: str, evidence_count: int) -> dict | None:
    keywords = extract_keywords(kiwi, sentence)
    candidates = search(conn, keywords, limit=50) if keywords else []
    if not candidates:
        return None

    top_chunks = rerank(sentence, candidates, top_k=evidence_count)
    prompt = build_prompt(sentence, top_chunks)
    raw_response = ask_ollama(ollama_url, model, prompt)
    verdict = parse_llm_response(raw_response)
    if verdict is None:
        return None  # 형식 오류 - 이 문장은 판단불가로 취급(§10), 결과에 포함하지 않는다
    if verdict["label"] != "반박":
        return None
    return build_claim(sentence, verdict, top_chunks)


def main() -> None:
    import sqlite3

    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wiki-index", required=True, type=Path)
    parser.add_argument("--ollama-url", required=True)
    parser.add_argument("--ollama-model", required=True)
    parser.add_argument("--evidence-count", type=int, default=5)
    args = parser.parse_args()

    sentences = json.loads(args.sentences.read_text(encoding="utf-8"))

    conn = sqlite3.connect(args.wiki_index)
    snapshot_row = conn.execute("SELECT value FROM meta WHERE key = 'snapshot'").fetchone()
    snapshot = snapshot_row[0] if snapshot_row else ""

    kiwi = Kiwi()
    claims = []
    for sentence in sentences:
        claim = judge_sentence(kiwi, conn, sentence, args.ollama_url, args.ollama_model, args.evidence_count)
        if claim is not None:
            claims.append(claim)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"model": args.ollama_model, "wiki_snapshot": snapshot, "claims": claims}
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
