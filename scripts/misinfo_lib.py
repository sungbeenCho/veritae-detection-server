"""LLM 프롬프트 조립과 응답 검증 - 순수 함수만 모아서 실제 모델/네트워크 없이 테스트한다.
프롬프트 문구는 2026-09-29 실측(위키 검색 포함 20건, 16/20)에 쓴 것과 동일하게 고정한다
(docs/superpowers/specs/2026-09-29-misinformation-detection-design.md §5) - 바꾸려면
tests/regression의 회귀 테스트셋을 반드시 다시 돌려야 한다.
"""
from __future__ import annotations

import json

from .wiki_index import wiki_url

VALID_LABELS = {"지지", "반박", "판단불가"}

PROMPT_TEMPLATE = """너는 사실 검증 도우미다. 아래 [근거 문단]들은 위키백과에서 자동으로 찾아온 조각이며, 주장과 관련 없는 조각이 섞여 있을 수 있다. [근거 문단]에 적힌 내용만 보고 [주장]을 판정하라. 네가 원래 알고 있는 지식은 쓰지 마라.
- 지지: 근거 문단이 주장이 참이라고 말한다
- 반박: 근거 문단이 주장이 거짓이라고 말한다
- 판단불가: 근거 문단이 주장에 대해 참/거짓을 말하지 않는다

[근거 문단]
{premise}

[주장]
{hypothesis}

label과 한 문장짜리 reason을 JSON으로 답하라."""


def build_prompt(sentence: str, evidence_chunks: list[tuple[str, str]]) -> str:
    premise = "\n".join(f"- ({title}) {text}" for title, text in evidence_chunks)
    return PROMPT_TEMPLATE.format(premise=premise, hypothesis=sentence)


def parse_llm_response(raw_response: str) -> dict | None:
    try:
        data = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        return None
    label = data.get("label")
    reason = data.get("reason")
    if label not in VALID_LABELS or not isinstance(reason, str) or not reason.strip():
        return None
    return {"label": label, "reason": reason}


def build_claim(sentence: str, verdict: dict, evidence_chunks: list[tuple[str, str]]) -> dict:
    return {
        "sentence": sentence,
        "reason": verdict["reason"],
        "evidence": [
            {"title": title, "text": text, "url": wiki_url(title)} for title, text in evidence_chunks
        ],
    }
