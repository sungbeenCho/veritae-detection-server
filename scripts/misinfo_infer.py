# scripts/misinfo_infer.py
"""문장 목록 -> 위키 검색 -> e5 재정렬 -> 앞뒤 문맥과 문서 첫 문단 붙이기 -> LLM 1차 판정 -> 반박이면
e5로 후보 문장 추리기 -> LLM이 근거 문장 선택 -> NLI 분류기로 모순 확인 -> 확인된 문장과 링크만 결과로.
text-extraction conda env(kiwipiepy, mwparserfromhell, torch, transformers 설치됨)에서
실행되어야 한다. veritae-detection-server(FastAPI)는 이 스크립트를 subprocess로 호출하고
--output 경로의 JSON만 읽는다 - scam_infer.py와 동일한 패턴.

docs(veritae-server 레포): docs/superpowers/specs/2026-09-29-misinformation-detection-design.md
"""
import argparse
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

from kiwipiepy import Kiwi

from misinfo_lib import (
    JUDGE_SCHEMA,
    SELECT_SCHEMA,
    build_claim,
    build_judge_prompt,
    build_select_prompt,
    fit_blocks,
    parse_judge_response,
    parse_select_response,
    replace_block_numbers,
)
from nli_check import ContradictionScorer, NliModel, candidate_sentences, confirm_sentences, default_model_path
from wiki_index import (
    Chunk,
    expand_with_neighbors,
    extract_keywords,
    get_snapshot,
    search,
    title_chunks,
    web_search_terms,
)

# 이름 있는 로거를 쓴다 - 회귀 테스트가 이 로거만 자세히 보여주고 httpx/HuggingFace 로그는 끌 수 있게.
logger = logging.getLogger("misinfo")

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


# 한 문서에서 고를 수 있는 조각 수. 상위 5개가 전부 한 문서('선풍기 사망설', '에펠탑의 레플리카…')
# 조각이라 다른 문서의 근거('미신' 문서의 "선풍기 미신" 등)가 밀려났다(2026-10-01 실측). 앞뒤 문맥을
# 붙이므로 한 문서에서 2개면 그 문서의 내용은 충분히 담긴다.
MAX_CHUNKS_PER_TITLE = 2


def rerank(sentence: str, candidates: list[Chunk], top_k: int) -> list[Chunk]:
    if not candidates:
        return []
    query_vec = embed(["query: " + sentence])
    passage_vecs = embed([f"passage: {c.title} {c.text}" for c in candidates])
    scores = (passage_vecs @ query_vec.T).squeeze(-1)
    return pick_diverse([candidates[i] for i in scores.argsort(descending=True).tolist()], top_k)


# 근거 문장 선택(LLM)에 보여줄 후보 문장 수. 검색 문단 전체를 문장으로 쪼개면 수십 개가 되는데,
# 4B 모델은 그중에서 "레플리카 에펠탑의 높이는 330 m이다", "고래상어의 수명은 70년" 같은 무관한
# 문장을 골라 맞는 반박을 버렸다(2026-10-01 데스크탑 실측: 놓친 19개 중 5개가 이 경우). e5로 주장과
# 가까운 문장만 먼저 추린다. e5만으로 고르면 "부산은 임시 수도가 되었다" 같은 틀린 근거가 섞여
# (같은 날 비교), 고르는 판단은 LLM에 남겨 둔다.
SELECT_SHORTLIST_SIZE = 5


def shortlist_sentences(claim: str, candidates: list[tuple[str, str]], size: int = SELECT_SHORTLIST_SIZE) -> list[tuple[str, str]]:
    """(제목, 문장) 후보 중 claim과 의미가 가까운 순으로 size개."""
    if len(candidates) <= size:
        return candidates
    query_vec = embed(["query: " + claim])
    passage_vecs = embed([f"passage: {sentence}" for _, sentence in candidates])
    scores = (passage_vecs @ query_vec.T).squeeze(-1)
    return [candidates[i] for i in scores.argsort(descending=True).tolist()[:size]]


def pick_diverse(ranked: list[Chunk], top_k: int, per_title: int = MAX_CHUNKS_PER_TITLE) -> list[Chunk]:
    """순위 순으로 고르되 한 문서에서는 per_title개까지만 고른다."""
    picked: list[Chunk] = []
    counts: dict[str, int] = {}
    for chunk in ranked:
        if counts.get(chunk.title, 0) < per_title:
            picked.append(chunk)
            counts[chunk.title] = counts.get(chunk.title, 0) + 1
            if len(picked) == top_k:
                break
    return picked


# 모델이 입력과 출력을 합쳐 한 번에 다룰 수 있는 토큰 수. 서버 기본값(데스크탑 실측 4096)에
# 기대지 않고 명시한다 - 입력이 이 한도를 넘으면 Ollama가 프롬프트 앞부분(지시문)을 조용히
# 잘라내고, 출력이 남은 한도를 넘으면 JSON을 다 쓰기 전에 멈춰 빈 응답이 된다(2026-10-01 실측:
# 생각 모드가 4096을 다 써서 done_reason=length, response=''). 입력 크기는 misinfo_lib.fit_blocks가
# 이 한도 안으로 보장한다.
OLLAMA_NUM_CTX = 8192


def ask_ollama(ollama_url: str, model: str, prompt: str, schema: dict) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "format": schema,
        "stream": False,
        # 생각 모드는 쓰지 않는다 - qwen3.5:4b는 생각이 수천 토큰으로 길어져 한도에 걸리거나
        # 문장당 수십 초가 걸려 분석 제한 시간(MISINFO_TIMEOUT_SECONDS)을 넘긴다(2026-10-01 실측).
        "think": False,
        "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX},
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
        logger.warning("Ollama 모델 언로드 요청 실패 (model=%s)", model, exc_info=True)


# 위키백과 검색 기능(무료 공개 API, 유료 LLM 아님). 로컬 키워드 검색은 단어가 다르면 문서를 못 찾는다 -
# "달 착륙은 조작되었다"로 "달착륙 음모론"을, "지구는 평평하다"로 "지평설"을 못 찾았다(2026-10-01 실측).
# 위키백과 검색은 이 둘을 찾았고, 반대로 로컬 검색이 찾는 문서(에펠탑, 후지산)는 못 찾아서 둘을 같이 쓴다.
# 제목만 받아오고 문서 내용은 로컬 인덱스의 것을 쓴다. 사용자 문장을 그대로 보내지 않고 핵심 단어만 보낸다.
# 외부 전송이라 설정(MISINFO_WIKI_SEARCH)으로 켤 때만 쓴다 - 기본은 꺼짐.
WIKI_SEARCH_URL = "https://ko.wikipedia.org/w/api.php"
WIKI_SEARCH_TITLES = 5
WIKI_SEARCH_TIMEOUT_SECONDS = 5
# 위키미디어 API 정책상 연락처가 담긴 User-Agent를 보내야 한다.
WIKI_USER_AGENT = "VeritaeMisinfo/1.0 (https://github.com/sungbeenCho/veritae-detection-server)"


def wiki_search_titles(terms: str, limit: int = WIKI_SEARCH_TITLES) -> list[str]:
    """위키백과 검색 상위 문서 제목. 실패하면(네트워크 등) 빈 목록 - 로컬 검색만으로 계속 판정한다."""
    if not terms.strip():
        return []
    query = urllib.parse.urlencode(
        {"action": "query", "list": "search", "format": "json", "srnamespace": 0, "srlimit": limit, "srsearch": terms}
    )
    req = urllib.request.Request(f"{WIKI_SEARCH_URL}?{query}", headers={"User-Agent": WIKI_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=WIKI_SEARCH_TIMEOUT_SECONDS) as resp:
            return [hit["title"] for hit in json.loads(resp.read())["query"]["search"]]
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("위키백과 검색 실패 - 로컬 검색만으로 판정함 (terms=%r)", terms, exc_info=True)
        return []


def judge_sentence(
    kiwi: Kiwi, conn, sentence: str, ollama_url: str, model: str, evidence_count: int, nli: ContradictionScorer,
    wiki_search: bool = False,
) -> dict | None:
    keywords = extract_keywords(kiwi, sentence)
    candidates = search(conn, keywords, limit=50) if keywords else []
    if wiki_search:
        seen = {c.rowid for c in candidates}
        for title in wiki_search_titles(web_search_terms(kiwi, sentence)):
            for chunk in title_chunks(conn, title, keywords):
                if chunk.rowid not in seen:
                    seen.add(chunk.rowid)
                    candidates.append(chunk)
    if not candidates:
        return None

    top_chunks = rerank(sentence, candidates, top_k=evidence_count)
    evidence_blocks = fit_blocks(expand_with_neighbors(conn, top_chunks))

    raw_response = ask_ollama(ollama_url, model, build_judge_prompt(sentence, evidence_blocks), JUDGE_SCHEMA)
    first = parse_judge_response(raw_response, len(evidence_blocks))
    if first is None:
        # 형식 오류 - 이 문장은 판단불가로 취급(§10), 결과에 포함하지 않는다. 다만 조용히
        # 넘어가면 LLM 출력이 계속 깨지고 있어도 알아챌 방법이 없으므로 경고 로그를 남긴다.
        logger.warning("LLM 응답 형식 오류로 문장을 판단불가 처리함: sentence=%r raw_response=%r", sentence, raw_response)
    logger.debug(
        "1차 판정: sentence=%r verdict=%r evidence_titles=%r",
        sentence, first, [title for title, _ in evidence_blocks],
    )
    if first is None or first["label"] != "반박" or not first["evidence_ids"]:
        return None

    # 반박은 사용자에게 "이 문장은 거짓"이라고 보여주는 유일한 판정이라, e5가 주장과 가까운 문장을
    # 추리고 그중에서 LLM이 근거 문장을 고르고(관련성), 고른 문장만 NLI가 모순인지 확인한다(판정).
    # 둘 다 통과한 문장만 근거로 표시하고, 하나도 없으면 반박을 버린다(misinfo_lib.py 모듈 설명 참고).
    # 후보는 1차 판정이 인용한 문단만이 아니라 검색된 문단 전체에서 고른다 - 1차 판정이 엉뚱한 문단을
    # 인용해 맞는 근거("학계의 중론은 … 자폐증 발생률을 증가시키지 않았으며")를 놓친 경우가 있었다
    # (2026-10-01 비교). 1차 판정은 반박 여부를 거르는 관문으로 쓴다.
    claim = first["core_claim"]
    candidates = shortlist_sentences(claim, candidate_sentences(evidence_blocks))
    if not candidates:
        return None
    raw_selection = ask_ollama(ollama_url, model, build_select_prompt(claim, candidates), SELECT_SCHEMA)
    picked_ids = parse_select_response(raw_selection, len(candidates))
    if picked_ids is None:
        logger.warning("근거 문장 선택 응답 형식 오류로 판단불가 처리함: sentence=%r raw_response=%r", sentence, raw_selection)
        return None
    picked = [candidates[i - 1] for i in picked_ids]
    confirmed = confirm_sentences(nli, picked, claim)
    logger.debug(
        "근거 문장 선택/NLI 확인: sentence=%r claim=%r picked=%r confirmed=%r",
        sentence, claim, picked, [(c["title"], c["sentence"], round(c["score"], 3)) for c in confirmed],
    )
    if not confirmed:
        logger.info("1차 반박의 근거 문장이 확인되지 않아 제외함: sentence=%r first_reason=%r", sentence, first["reason"])
        return None
    # 1차 판정의 번호는 evidence_blocks 기준이다.
    reason = replace_block_numbers(first["reason"], evidence_blocks)
    # 화면에는 반박 문장 하나와 문서 링크만 보여준다 - 판정에 쓴 긴 문맥(앞뒤 문단)을 그대로 내보내면
    # 핵심 문장이 무관한 내용 사이에 묻힌다(2026-10-01 사용자 결정).
    return build_claim(sentence, reason, [(c["title"], c["sentence"]) for c in confirmed])


def main() -> None:
    import sqlite3

    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wiki-index", required=True, type=Path)
    parser.add_argument("--ollama-url", required=True)
    parser.add_argument("--ollama-model", required=True)
    parser.add_argument("--evidence-count", type=int, default=5)
    parser.add_argument("--nli-model", default=default_model_path())
    parser.add_argument("--wiki-search", action="store_true", help="위키백과 검색 기능도 함께 쓴다(핵심 단어만 외부 전송)")
    args = parser.parse_args()

    sentences = json.loads(args.sentences.read_text(encoding="utf-8"))
    nli = NliModel(args.nli_model)

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
            claim = judge_sentence(
                kiwi, conn, sentence, args.ollama_url, args.ollama_model, args.evidence_count, nli.contradiction_scores,
                wiki_search=args.wiki_search,
            )
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
