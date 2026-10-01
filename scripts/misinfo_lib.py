"""LLM 프롬프트 조립과 응답 검증 - 순수 함수만 모아서 실제 모델/네트워크 없이 테스트한다.
프롬프트와 응답 형식은 tests/regression 회귀 테스트셋으로 검증한 그대로 고정한다
(docs/superpowers/specs/2026-09-29-misinformation-detection-design.md §5) - 바꾸려면 데스크탑에서
회귀 테스트셋을 반드시 다시 돌려야 한다.

판정은 세 단계다(사실검증 표준 구조인 FEVER의 "문서 검색 -> 근거 문장 선택 -> 판정"을 따른다).
1) 1차 판정(LLM): 핵심 주장을 정리하고 반박 후보와 인용 문단을 고른다.
2) 근거 문장 선택: 검색된 문단을 문장으로 쪼개 e5로 주장과 가까운 5개를 추리고, LLM이 그중에서 주장이
   거짓임을 보여주는 문장 번호만 고른다(misinfo_infer.shortlist_sentences).
3) 확인(nli_check.py의 NLI 분류기): 2)에서 고른 문장만 주장과 모순인지 확인한다.
2)를 빼고 NLI가 모순 점수가 가장 높은 문장을 근거로 고르게 했더니, NLI는 주어만 같고 무관한 문장에도 높은
점수를 줘서 "지구의 받침대 베어링" 같은 문장이 반박 근거로 표시됐다(2026-10-01 데스크탑 실측). FEVER 연구도
문장 선택 단계를 빼면 정확도가 약 10%p 떨어진다고 보고한다.
"""
from __future__ import annotations

import json
import re

from wiki_index import wiki_url

JUDGE_LABELS = ("지지", "반박", "판단불가")

# 필드 순서가 곧 모델이 생각하는 순서다. 모델은 앞에서부터 한 글자씩 쓰므로, 판정을 먼저
# 쓰게 하면 근거를 따지기 전에 판정부터 정하고 이유는 그 뒤에 끼워 맞춘다 - 2026-10-01
# 실측에서 "label=반박인데 reason은 '일치합니다'"가 나온 원인이다. 판정은 항상 맨 뒤에 둔다.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "core_claim": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
        "label": {"type": "string", "enum": list(JUDGE_LABELS)},
    },
    "required": ["core_claim", "evidence_ids", "reason", "label"],
}

# 근거 문장 선택: 왜 그 문장인지(analysis)를 먼저 쓰고 번호는 맨 뒤에 고른다(판정을 먼저 쓰지 않게 하는 것과 같은 이유).
SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "sentence_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["analysis", "sentence_ids"],
}

_GROUNDING_RULE = "[근거 문단]에 적힌 내용만 보고 판정하라. 네가 원래 알고 있는 지식은 쓰지 마라."

_CORE_CLAIM_RULE = (
    "[주장]에서 참/거짓을 따질 핵심 내용을 한 문장으로 적는다. [주장]이 \"~라는 이야기를 들었다\", "
    "\"~라고 한다\", \"~라더라\"처럼 전해 들은 말을 옮기는 문장이면, 전해 들은 그 내용 자체를 적는다. "
    "반대로 \"~라는 속설이 있다\", \"~라는 음모론이 있다\", \"~라는 것은 잘못 알려진 것이다\"처럼 그 내용을 "
    "속설이나 틀린 이야기라고 소개하는 문장이면, 문장 그대로를 적는다. 글자 인식 오류로 보이는 오타는 바로잡는다."
)

_REASON_RULE = (
    "reason은 사용자에게 그대로 보여주는 문장인데 사용자는 문단 번호를 볼 수 없다. "
    "\"[1]\", \"문단 2\" 같은 번호는 쓰지 말고 문서 제목으로 가리킨다."
)

_REFUTES = "근거 문단의 내용이 사실이라면 core_claim은 거짓일 수밖에 없다(core_claim을 직접 부정하지 않아도 된다)"

# 소개와 평가를 구분한다. 위키는 속설을 "사실이 아니다"보다 "과학적 근거가 없는 미신이다"처럼
# 평가하는 경우가 많은데, "근거가 없다"는 엄밀히는 "거짓이다"와 달라서 이 규칙이 없으면 검증이
# 대표 사례(선풍기 사망설)조차 반박 아님으로 떨어뜨릴 수 있다.
_MENTION_RULE = (
    "어떤 내용을 \"~라는 가설이 있다\", \"~라는 속설이 있다\", \"~라는 주장이 있다\"고 소개만 하는 문단은, "
    "그 내용 자체가 참이라는 근거도 거짓이라는 근거도 아니다. 반대로 그 내용을 \"근거 없는 속설\", \"미신\", "
    "\"잘못 알려진 것\", \"사실이 아니다\"라고 평가하는 문단은, 그 내용이 거짓이라는 근거다."
)

# 정의만으로는 4B 모델이 "직접 부정하지 않았다"며 문자 그대로만 따져 맞는 반박을 버렸다(2026-10-01
# 실측: "에펠탑은 프랑스 파리에 있다"로 "독일에 있다"를 반박하지 못함). 예시로 기준을 보여준다.
# 예시는 회귀 테스트셋 문장과 겹치지 않게 골랐다 - 시험 문제의 답을 가르치면 실측이 무의미해진다.
_EXAMPLES = """판정 예시(아래 [근거 문단]과는 관계없는 예시다):
- 근거 "불국사는 경상북도 경주시에 있다." / 주장 "불국사는 전라남도에 있다." → 반박이다. 한 절이 두 지역에 동시에 있을 수 없다.
- 근거 "모나리자는 레오나르도 다 빈치가 그린 그림이다." / 주장 "모나리자는 미켈란젤로가 그렸다." → 반박이다.
- 근거 "혈액형으로 성격을 알 수 있다는 것은 과학적 근거가 없는 속설이다." / 주장 "혈액형으로 성격을 알 수 있다." → 반박이다. 근거 없는 속설이라고 평가한다.
- 근거 "보름달이 뜨면 범죄가 늘어난다는 속설이 있다." / 주장 "보름달이 뜨면 범죄가 늘어난다." → 반박이 아니다. 속설을 소개만 한다.
- 근거 "창덕궁은 조선의 궁궐이다." / 주장 "창덕궁은 종로구에 있다." → 반박이 아니다. 근거가 위치를 말하지 않는다."""

_CAREFUL_RULE = (
    "근거 문단은 하나씩 따로 따진다. 주장과 관련 없는 문단은 무시하고, 어느 한 문단이라도 주장을 거짓으로 "
    "만들면 그것으로 충분하다. 이름이 비슷한 다른 인물·장소·시대를 같은 것으로 착각하지 않도록 이름을 정확히 비교한다."
)

JUDGE_TEMPLATE = f"""너는 사실 검증 도우미다. 아래 [근거 문단]들은 위키백과에서 자동으로 찾아온 것이며, 주장과 관련 없는 문단이 섞여 있을 수 있다. {_GROUNDING_RULE} {_CAREFUL_RULE}

다음 순서대로 답하라.
1. core_claim: {_CORE_CLAIM_RULE}
2. evidence_ids: core_claim이 참인지 거짓인지를 알려주는 근거 문단의 번호를 모두 적는다. 그런 문단이 없으면 빈 목록으로 둔다.
3. reason: 고른 근거 문단이 core_claim에 대해 무엇이라고 말하는지 한 문장으로 설명한다. {_REASON_RULE}
4. label: reason을 바탕으로 판정한다.
- 지지: 고른 근거 문단이 core_claim이 참이라고 말한다
- 반박: {_REFUTES}
- 판단불가: 근거 문단이 core_claim의 참/거짓을 정해주지 않는다
{_MENTION_RULE}

{_EXAMPLES}

[근거 문단]
{{premise}}

[주장]
{{hypothesis}}"""

# 근거 문장 선택용 규칙은 1차 판정 규칙과 뜻은 같고 가리키는 단위만 "문장"이다. 1차 판정 문구는 회귀
# 테스트로 맞춘 그대로 두기 위해 따로 적는다.
_SELECT_MENTION_RULE = (
    "어떤 내용을 \"~라는 가설이 있다\", \"~라는 속설이 있다\", \"~라는 주장이 있다\"고 소개만 하는 문장은 "
    "고르지 않는다. 그 내용을 \"근거 없는 속설\", \"미신\", \"잘못 알려진 것\", \"사실이 아니다\"라고 평가하는 "
    "문장은 고른다."
)

# 문장 하나씩 떼어 보면 주어만 같은 무관한 문장이 반박처럼 보이기 쉽다(2026-10-01: "지구는 평평하다"에
# "지구의는 받침대에 베어링이 있다"). 마지막 예시가 그 경우다. 예시는 회귀 테스트셋 문장과 겹치지 않게 골랐다.
_SELECT_EXAMPLES = """고르는 예시(아래 [문장]과는 관계없는 예시다):
- 주장 "불국사는 전라남도에 있다." / 문장 "불국사는 경상북도 경주시에 있다." → 고른다. 한 절이 두 지역에 동시에 있을 수 없다.
- 주장 "모나리자는 미켈란젤로가 그렸다." / 문장 "모나리자는 레오나르도 다 빈치가 그린 그림이다." → 고른다.
- 주장 "혈액형으로 성격을 알 수 있다." / 문장 "이는 과학적 근거가 없는 속설이다." → 고른다. 근거 없는 속설이라고 평가한다.
- 주장 "보름달이 뜨면 범죄가 늘어난다." / 문장 "보름달이 뜨면 범죄가 늘어난다는 속설이 있다." → 고르지 않는다. 속설을 소개만 한다.
- 주장 "불국사는 전라남도에 있다." / 문장 "불국사의 다보탑은 10원 동전에 새겨져 있다." → 고르지 않는다. 같은 대상이 나오지만 위치와는 관계없다."""

SELECT_TEMPLATE = f"""너는 사실 검증 도우미다. 아래 [문장]들은 위키백과에서 가져온 것이고, 괄호 안은 문서 제목이다. 이 중에서 [주장]이 거짓이라는 것을 보여주는 문장의 번호를 고르라. [문장]에 적힌 내용만 보고 고르라. 네가 원래 알고 있는 지식은 쓰지 마라.

다음 순서대로 답하라.
1. analysis: 고른 문장이 [주장]의 어느 부분과 어떻게 맞지 않는지 짧게 적는다.
2. sentence_ids: 고른 문장 번호를 모두 적는다. 그런 문장이 없으면 빈 목록으로 둔다.

고를 문장: 그 문장의 내용이 사실이라면 [주장]은 거짓일 수밖에 없는 문장([주장]을 직접 부정하지 않아도 된다). [주장]과 같은 이름이나 대상이 나오더라도, [주장]이 참인지 거짓인지와 관계없는 내용이면 고르지 않는다. 이름이 비슷한 다른 인물·장소·시대를 같은 것으로 착각하지 않도록 이름을 정확히 비교한다. {_SELECT_MENTION_RULE}

{_SELECT_EXAMPLES}

[문장]
{{sentences}}

[주장]
{{claim}}"""


# 근거 문단 전체 글자 수 상한. 입력이 모델 컨텍스트(misinfo_infer.OLLAMA_NUM_CTX=8192 토큰)를 넘으면
# Ollama가 프롬프트 앞부분 - 오반박을 막는 지시문 - 부터 조용히 잘라낸다(2026-10-01 리뷰). 한국어는
# 한 토큰이 한 글자보다 짧지 않으므로, 근거 4,500자 + 지시문·예시 약 2,100자 + 주장 + 출력이 어떤
# 토크나이저에서도 8192 안에 들어간다. 보통 입력(문단 5개 x 약 600자)은 이 상한에 걸리지 않는다.
PREMISE_MAX_CHARS = 4500


def fit_blocks(blocks: list[tuple[str, str]], max_chars: int = PREMISE_MAX_CHARS) -> list[tuple[str, str]]:
    """순위 순으로 문단을 담다가 상한을 넘는 문단은 뺀다. 문단 중간은 자르지 않는다 - 잘라내면
    "가설이 있다" 뒤의 "근거 없다"가 빠지는 문제(2026-10-01)가 다시 생긴다. 1순위 문단 하나가
    혼자 상한을 넘는 극단적인 경우에만 그 문단을 잘라서 넣는다."""
    kept: list[tuple[str, str]] = []
    used = 0
    for title, text in blocks:
        size = len(title) + len(text)
        if used + size <= max_chars:
            kept.append((title, text))
            used += size
        elif not kept:
            kept.append((title, text[: max_chars - len(title)]))
            used = max_chars
    return kept


def replace_block_numbers(reason: str, blocks: list[tuple[str, str]]) -> str:
    """이유 문장의 "[1]" 같은 문단 번호를 문서 제목으로 바꾼다. 사용자는 번호를 볼 수 없는데,
    프롬프트로 금지해도 모델이 자주 어겼다(2026-10-01 실측: "[1] 문서에서 ..."). 범위 밖 번호는 둔다."""
    def to_title(match: re.Match) -> str:
        number = int(match.group(1))
        return f"'{blocks[number - 1][0]}'" if 1 <= number <= len(blocks) else match.group(0)

    return re.sub(r"\[(\d+)\]", to_title, reason)


def _format_blocks(blocks: list[tuple[str, str]]) -> str:
    return "\n".join(f"[{i}] ({title}) {text}" for i, (title, text) in enumerate(blocks, start=1))


def build_select_prompt(claim: str, candidates: list[tuple[str, str]]) -> str:
    """candidates: (문서 제목, 문장) 목록. 번호는 1부터."""
    numbered = "\n".join(f"[{i}] ({title}) {sentence}" for i, (title, sentence) in enumerate(candidates, start=1))
    return SELECT_TEMPLATE.format(sentences=numbered, claim=claim)


def parse_select_response(raw_response: str, candidate_count: int) -> list[int] | None:
    """형식이 틀리면 None, 고른 문장이 없으면 빈 목록."""
    data = _load_object(raw_response)
    if data is None or not isinstance(data.get("analysis"), str) or not isinstance(data.get("sentence_ids"), list):
        return None
    return _valid_ids(data["sentence_ids"], candidate_count)


def build_judge_prompt(sentence: str, evidence_blocks: list[tuple[str, str]]) -> str:
    return JUDGE_TEMPLATE.format(premise=_format_blocks(evidence_blocks), hypothesis=sentence)


def _load_object(raw_response: str) -> dict | None:
    try:
        data = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _valid_ids(values: list, count: int) -> list[int]:
    """1..count 범위의 정수만 순서대로 중복 없이 남긴다(bool은 int의 하위 타입이라 따로 뺀다)."""
    ids: list[int] = []
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= count and value not in ids:
            ids.append(value)
    return ids


def parse_judge_response(raw_response: str, evidence_count: int) -> dict | None:
    """형식이 틀리면 None."""
    data = _load_object(raw_response)
    if data is None:
        return None
    core_claim = data.get("core_claim")
    evidence_ids = data.get("evidence_ids")
    reason = data.get("reason")
    label = data.get("label")
    # 이유는 반박일 때만 쓰인다 - 판단불가/지지에서 이유가 비어 있는 건 형식 오류로 보지 않는다
    # (2026-10-01 실측: "편의점" 문장이 판단불가 + 빈 이유로 와서 불필요한 경고가 났다).
    if (
        label not in JUDGE_LABELS
        or not isinstance(reason, str)
        or (label == "반박" and not reason.strip())
        or not isinstance(core_claim, str)
        or not core_claim.strip()
        or not isinstance(evidence_ids, list)
    ):
        return None
    return {
        "core_claim": core_claim,
        "evidence_ids": _valid_ids(evidence_ids, evidence_count),
        "reason": reason,
        "label": label,
    }


def build_claim(sentence: str, reason: str, evidence_blocks: list[tuple[str, str]]) -> dict:
    """evidence_blocks는 NLI 확인을 통과한(모순 문장이 실제로 든) 문단만 넘긴다 - 검색 후보 전부가 아니다."""
    return {
        "sentence": sentence,
        "reason": reason,
        "evidence": [{"title": title, "text": text, "url": wiki_url(title)} for title, text in evidence_blocks],
    }
