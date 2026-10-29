# scripts/check_scam_data_surface.py
"""사기감지 학습 데이터 겉모양 점검 - 내용 없이 겉모양만으로 사기/정상 라벨이 맞혀지는지 본다.

Lilju 대체 분류기 데이터(tests/scam/*.json, 지침 tests/scam/SPEC.md)를 학습하기 전에 돌린다. 겉모양(길이, 문장 수,
문장부호, 숫자, 존댓말 끝맺음)이 라벨과 엮여 있으면 모델이 사기 '행위'가 아니라 작성 습관을 배운다(KorCCVi가
출처만으로 라벨이 맞혀진 것과 같은 실패). 분류기 학습은 하지 않기로 했지만(2026-10-27, LLM 판정으로 대체) 시험 세트 점검에 그대로 쓸 수 있다.

판정 규칙이 아니라 데이터 점검용이다. numpy만 쓴다.
사용: python check_scam_data_surface.py ..\\tests\\scam\\pilot_A.json ..\\tests\\scam\\pilot_B.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

FEATURES = ["글자 수", "문장 수", "문장 평균 길이", "물음표 수", "느낌표 수", "숫자 글자 수", "존댓말 끝맺음 비율"]
POLITE_ENDING = re.compile(r"(요|니다|니까|세요|십시오)[.?!~]*$")
BOOTSTRAP = 2000


def features(conversation: dict) -> list[float]:
    sentences = [s["text"] for s in conversation["sentences"]]
    text = " ".join(sentences)
    count = len(sentences)
    return [
        len(text),
        count,
        len(text) / count,
        text.count("?"),
        text.count("!"),
        sum(ch.isdigit() for ch in text),
        sum(bool(POLITE_ENDING.search(s.strip())) for s in sentences) / count,
    ]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """사기=1이 정상=0보다 점수가 높을 확률(동점 0.5). 0.5에서 멀수록(위든 아래든) 겉모양이 라벨을 가른다."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 2000, lr: float = 0.1, l2: float = 0.1) -> np.ndarray:
    w = np.zeros(x.shape[1] + 1)
    xb = np.hstack([x, np.ones((len(x), 1))])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-xb @ w))
        w -= lr * (xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0])
    return w


def leave_one_pair_out(x: np.ndarray, y: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """짝 하나씩 빼고 나머지로 학습해 뺀 짝을 채점한다(짝을 학습·평가에 나눠 넣지 않는 것과 같은 규칙)."""
    scores = np.zeros(len(y))
    for pair in np.unique(pairs):
        test = pairs == pair
        mean, std = x[~test].mean(0), x[~test].std(0) + 1e-9
        w = fit_logistic((x[~test] - mean) / std, y[~test])
        xt = (x[test] - mean) / std
        scores[test] = 1 / (1 + np.exp(-(np.hstack([xt, np.ones((len(xt), 1))]) @ w)))
    return scores


def pair_bootstrap_ci(scores: np.ndarray, y: np.ndarray, pairs: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(pairs)
    values = []
    for _ in range(BOOTSTRAP):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(pairs == p) for p in picked])
        if len(set(y[idx])) == 2:
            values.append(auc(scores[idx], y[idx]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def words(conversation: dict) -> set[str]:
    text = " ".join(s["text"] for s in conversation["sentences"])
    return {w for w in re.sub(r"[^0-9A-Za-z가-힣]+", " ", text).split() if len(w) > 1}


def bag_of_words_check(rows: list[dict], y: np.ndarray, groups: np.ndarray) -> None:
    """요소 목록 밖 지름길 점검(3판 지침): 단어 있음/없음만으로 학습한 단순 모델의 AUC와, 라벨을 가장 강하게 가르는 단어.
    사기 행위를 나타내는 말이 아니라 말투·형식 단어가 위에 오면 작성 습관이 지름길이 된 것이다."""
    vocab = sorted({w for r in rows for w in words(r)})
    counts = {w: sum(w in words(r) for r in rows) for w in vocab}
    vocab = [w for w in vocab if counts[w] >= 2]
    x = np.array([[w in words(r) for w in vocab] for r in rows], dtype=float)
    scores = leave_one_pair_out(x, y, groups)
    low, high = pair_bootstrap_ci(scores, y, groups)
    print(f"\n[단어 있음/없음 단순 모델, 하나씩 빼고 평가] AUC {auc(scores, y):.2f} (95% 구간 {low:.2f}~{high:.2f}), 단어 {len(vocab)}개")
    scam_rate = {w: (x[y == 1, i].sum(), x[y == 0, i].sum()) for i, w in enumerate(vocab)}
    lean = sorted(vocab, key=lambda w: abs(scam_rate[w][0] - scam_rate[w][1]), reverse=True)[:25]
    print("  라벨이 가장 갈리는 단어(사기 대화 수 / 정상 대화 수): " + ", ".join(f"{w} {int(scam_rate[w][0])}/{int(scam_rate[w][1])}" for w in lean))


def main() -> None:
    rows = [row for path in sys.argv[1:] for row in json.loads(Path(path).read_text(encoding="utf-8"))]
    y = np.array([1 if r["label"] == "scam" else 0 for r in rows])
    # 2판까지는 짝(pair_id), 3판부터는 짝이 없어 대화(id) 단위로 묶는다.
    pairs = np.array([r.get("pair_id", r.get("id")) for r in rows])
    x = np.array([features(r) for r in rows], dtype=float)
    print(f"대화 {len(rows)}개 (사기 {y.sum()}, 정상 {len(y) - y.sum()}), 짝 {len(np.unique(pairs))}개")

    print("\n[작성자·주제별 사기 비율]")
    for key in ("writer", "topic", "element", "medium", "ambiguous"):
        groups = {}
        for r, label in zip(rows, y):
            if key in r or key == "ambiguous":
                groups.setdefault(str(r.get(key, False)), []).append(label)
        if not groups:
            continue
        print(f"  {key}: " + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(groups.items())))

    print("\n[겉모양 특징 하나씩] AUC(0.5=구분 못함), 짝 안에서 사기 쪽이 더 큰 짝 수 / 더 작은 짝 수")
    by_pair = {}
    for i, p in enumerate(pairs):
        by_pair.setdefault(p, {})[y[i]] = x[i]
    complete = [v for v in by_pair.values() if 0 in v and 1 in v]
    for j, name in enumerate(FEATURES):
        bigger = sum(v[1][j] > v[0][j] for v in complete)
        smaller = sum(v[1][j] < v[0][j] for v in complete)
        print(f"  {name}: AUC {auc(x[:, j], y):.2f}, 사기 쪽이 큼 {bigger} / 작음 {smaller} (짝 {len(complete)})")

    scores = leave_one_pair_out(x, y, pairs)
    low, high = pair_bootstrap_ci(scores, y, pairs)
    print(f"\n[겉모양 특징 전체로 학습한 단순 모델, 짝 하나씩 빼고 평가] AUC {auc(scores, y):.2f} (95% 구간 {low:.2f}~{high:.2f})")
    print("  판단 기준(설계서 4.3): 구간 하한이 0.5를 넘으면 겉모양만으로 라벨이 맞혀진다 -> 데이터를 다시 만든다.")
    bag_of_words_check(rows, y, pairs)


if __name__ == "__main__":
    main()
