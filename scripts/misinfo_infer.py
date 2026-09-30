# scripts/misinfo_infer.py
"""문장 목록 -> 위키 검색 -> e5 재정렬 -> LLM 판정 -> 반박된 문장만 결과로.
text-extraction conda env(kiwipiepy, mwparserfromhell, torch, transformers 설치됨)에서
실행되어야 한다. veritae-detection-server(FastAPI)는 이 스크립트를 subprocess로 호출하고
--output 경로의 JSON만 읽는다 - scam_infer.py와 동일한 패턴.

docs(veritae-server 레포): docs/superpowers/specs/2026-09-29-misinformation-detection-design.md
"""
import argparse
import json
import logging
import urllib.request
from pathlib import Path

from kiwipiepy import Kiwi

from misinfo_lib import build_claim, build_prompt, parse_llm_response
from wiki_index import extract_keywords, get_snapshot, search

OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["지지", "반박", "판단불가"]},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
}

EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-small"

# 프로세스당 한 번만 로드하는 e5 토크나이저/모델 캐시. embed()가 호출될 때마다
# from_pretrained로 다시 불러오면 rerank()가 문장당 두 번(질의+후보) 호출하므로
# 요청당 모델 로드가 2N번 일어나 MISINFO_TIMEOUT_SECONDS를 넘길 위험이 커진다
# (2026-09-30 리뷰). 모듈 전역에 캐싱하고, CUDA가 있으면 GPU로 옮긴다(스펙 §6/§14).
_embed_tokenizer = None
_embed_model = None
_embed_device = "cpu"


def _load_embed_model():
    global _embed_tokenizer, _embed_model, _embed_device
    if _embed_model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        _embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_ID)
        _embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_ID)
        _embed_model.eval()
        _embed_device = "cuda" if torch.cuda.is_available() else "cpu"
        _embed_model.to(_embed_device)
    return _embed_tokenizer, _embed_model, _embed_device


def embed(texts: list[str]):
    import torch

    tokenizer, model, device = _load_embed_model()
    batch = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    vectors = (hidden * mask).sum(1) / mask.sum(1)
    return torch.nn.functional.normalize(vectors, dim=-1).cpu()


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


def unload_ollama_model(ollama_url: str, model: str) -> None:
    """요청 처리가 끝나면 모델을 GPU에서 즉시 내린다(스펙 §5 [확정]: "요청 하나 처리하는
    동안만 모델을 올려두고 끝나면 내린다"). Ollama 기본 keep_alive(5분)를 그대로 두면
    GPU 큐 자리를 반환한 뒤에도 qwen3.5:4b가 VRAM에 남아 바로 다음 Whisper/SPAI 작업과
    메모리를 다투게 되어 GPU 큐를 만든 목적(메모리 부족 방지) 자체가 깨진다(2026-09-30 리뷰).
    main()이 도중에 실패해도 반드시 실행되도록 finally에서만 호출한다.
    """
    body = {"model": model, "keep_alive": 0}
    req = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except Exception:
        # 언로드 실패는 이번 요청 자체를 실패시킬 이유가 아니다(best-effort) - 다만 조용히
        # 삼키면 다음에도 GPU에 모델이 남아있는 원인을 못 찾으니 반드시 경고 로그를 남긴다.
        logging.warning("Ollama 모델 언로드 요청 실패 (model=%s)", model, exc_info=True)


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
        # 형식 오류 - 이 문장은 판단불가로 취급(§10), 결과에 포함하지 않는다. 다만 조용히
        # 넘어가면 LLM 출력이 계속 깨지고 있어도 알아챌 방법이 없으므로 경고 로그를 남긴다
        # (계획 Global Constraints 19행, 스펙 §10, 2026-09-30 리뷰).
        logging.warning(
            "LLM 응답 형식 오류로 문장을 판단불가 처리함: sentence=%r raw_response=%r",
            sentence, raw_response,
        )
        return None
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

    # 읽기 전용으로 연다 - 일반 sqlite3.connect(path)는 파일이 없으면 빈 DB 파일을 새로
    # 만들어버려서(스키마 없는 깨진 wiki_index.sqlite3), 이후 모든 검색이 "그냥 결과 없음"으로
    # 조용히 실패해 위키 인덱스 미설정 문제를 알아채기 어렵다(2026-09-30 리뷰). mode=ro로 열면
    # 파일이 없거나 인덱스 스키마가 없을 때 즉시 에러가 나서 misinfo_runner.py가 502로 드러낸다.
    # sqlite3 URI는 슬래시(/) 구분자를 쓴다 - Windows 경로(C:\...)를 그대로 넣으면
    # 역슬래시가 URI 이스케이프 문자로 오인될 수 있어 as_posix()로 변환한다.
    wiki_index_uri = f"file:{args.wiki_index.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(wiki_index_uri, uri=True)
    snapshot = get_snapshot(conn)

    kiwi = Kiwi()
    claims = []
    try:
        for sentence in sentences:
            claim = judge_sentence(kiwi, conn, sentence, args.ollama_url, args.ollama_model, args.evidence_count)
            if claim is not None:
                claims.append(claim)
    finally:
        # 도중에 예외가 나도(Ollama 호출 실패 등) 반드시 모델을 내린다(스펙 §5, 2026-09-30 리뷰).
        unload_ollama_model(args.ollama_url, args.ollama_model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"model": args.ollama_model, "wiki_snapshot": snapshot, "claims": claims}
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
