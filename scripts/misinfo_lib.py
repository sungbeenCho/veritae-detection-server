"""LLM 프롬프트 조립과 응답 검증 - 순수 함수만 모아서 실제 모델/네트워크 없이 테스트한다.
프롬프트와 응답 형식(OLLAMA_SCHEMA)은 tests/regression 회귀 테스트셋으로 검증한 그대로 고정한다
(docs/superpowers/specs/2026-09-29-misinformation-detection-design.md §5) - 바꾸려면 데스크탑에서
회귀 테스트셋을 반드시 다시 돌려야 한다.
"""
from __future__ import annotations

import json

from wiki_index import wiki_url

VALID_LABELS = {"지지", "반박", "판단불가"}

# 필드 순서가 곧 모델이 생각하는 순서다. 모델은 앞에서부터 한 글자씩 쓰므로, label을 먼저
# 쓰게 하면 근거를 따지기 전에 판정부터 정하고 reason은 그 뒤에 끼워 맞춘다 - 2026-10-01
# 실측에서 "label=반박인데 reason은 '일치합니다'"가 나온 원인이다. 핵심 주장 정리 → 근거 선택
# → 이유 → 판정 순서로 강제한다.
OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {
        "core_claim": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
        "label": {"type": "string", "enum": ["지지", "반박", "판단불가"]},
    },
    "required": ["core_claim", "evidence_ids", "reason", "label"],
}

PROMPT_TEMPLATE = """너는 사실 검증 도우미다. 아래 [근거 문단]들은 위키백과에서 자동으로 찾아온 것이며, 주장과 관련 없는 문단이 섞여 있을 수 있다. [근거 문단]에 적힌 내용만 보고 판정하라. 네가 원래 알고 있는 지식은 쓰지 마라.

다음 순서대로 답하라.
1. core_claim: [주장]에서 참/거짓을 따질 핵심 내용을 한 문장으로 적는다. [주장]이 "~라는 이야기를 들었다", "~라고 한다", "~라더라"처럼 전해 들은 말을 옮기는 문장이면, 전해 들은 그 내용 자체를 적는다. 반대로 "~라는 속설이 있다", "~라는 음모론이 있다", "~라는 것은 잘못 알려진 것이다"처럼 그 내용을 속설이나 틀린 이야기라고 소개하는 문장이면, 문장 그대로를 적는다. 글자 인식 오류로 보이는 오타는 바로잡는다.
2. evidence_ids: core_claim이 참인지 거짓인지를 직접 말해주는 근거 문단의 번호를 모두 적는다. 그런 문단이 없으면 빈 목록으로 둔다.
3. reason: 고른 근거 문단이 core_claim에 대해 무엇이라고 말하는지 한 문장으로 설명한다. 문단 번호 대신 문서 제목으로 가리킨다.
4. label: reason을 바탕으로 판정한다.
- 지지: 고른 근거 문단이 core_claim이 참이라고 말한다
- 반박: 고른 근거 문단이 core_claim이 거짓이라고 말한다
- 판단불가: 근거 문단이 core_claim의 참/거짓을 말하지 않는다
어떤 내용을 "~라는 가설이 있다", "~라는 속설이 있다", "~라는 주장이 있다"고 소개만 하는 문단은, 그 내용 자체가 참이라는 근거도 거짓이라는 근거도 아니다.

[근거 문단]
{premise}

[주장]
{hypothesis}"""


def build_prompt(sentence: str, evidence_blocks: list[tuple[str, str]]) -> str:
    premise = "\n".join(f"[{i}] ({title}) {text}" for i, (title, text) in enumerate(evidence_blocks, start=1))
    return PROMPT_TEMPLATE.format(premise=premise, hypothesis=sentence)


def parse_llm_response(raw_response: str, evidence_count: int) -> dict | None:
    """형식이 틀리면 None. evidence_ids는 1..evidence_count 범위의 정수만 순서대로 중복 없이 남긴다."""
    try:
        data = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    core_claim = data.get("core_claim")
    evidence_ids = data.get("evidence_ids")
    reason = data.get("reason")
    label = data.get("label")
    if (
        label not in VALID_LABELS
        or not isinstance(reason, str)
        or not reason.strip()
        or not isinstance(core_claim, str)
        or not core_claim.strip()
        or not isinstance(evidence_ids, list)
    ):
        return None
    valid_ids: list[int] = []
    for evidence_id in evidence_ids:
        if (
            isinstance(evidence_id, int)
            and not isinstance(evidence_id, bool)
            and 1 <= evidence_id <= evidence_count
            and evidence_id not in valid_ids
        ):
            valid_ids.append(evidence_id)
    return {"core_claim": core_claim, "evidence_ids": valid_ids, "reason": reason, "label": label}


def build_claim(sentence: str, verdict: dict, evidence_blocks: list[tuple[str, str]]) -> dict:
    """판정에 실제로 인용된 근거 문단(evidence_ids)만 근거로 담는다 - 검색 후보 전부가 아니다."""
    cited = [evidence_blocks[i - 1] for i in verdict["evidence_ids"]]
    return {
        "sentence": sentence,
        "reason": verdict["reason"],
        "evidence": [{"title": title, "text": text, "url": wiki_url(title)} for title, text in cited],
    }
