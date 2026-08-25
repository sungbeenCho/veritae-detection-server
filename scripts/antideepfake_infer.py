#!/usr/bin/env python
"""antideepfake_infer.py

AntiDeepfake(mms_300m 체크포인트)로 오디오 파일 하나를 추론해서, 전체 score와
시간 구간별(frame-level) evidence를 JSON으로 저장한다.

`antideepfake` conda env(fairseq 등 설치됨)에서 실행되어야 한다.
veritae-detection-server(FastAPI, detection-api env)는 이 스크립트를 subprocess로
호출하고 --output 경로의 JSON만 읽는다 - 무거운 의존성을 FastAPI 프로세스에
넣지 않기 위함 (SPAI 연동과 동일한 패턴).

docs(veritae-server 레포): docs/superpowers/specs/2026-08-18-audio-ai-detection-design.md §3, §4
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio

SAMPLE_RATE = 16000
# mms_300m의 wav2vec2 계열 SSL 프론트엔드는 16kHz 기준 320 샘플(20ms)마다 프레임 하나를
# 뽑는다(wav2vec2 계열 컨볼루션 feature encoder의 공통 stride) - 실제 체크포인트로 검증 필요.
FRAME_STRIDE_SECONDS = 0.02
EVIDENCE_SCORE_THRESHOLD = 0.5
MAX_EVIDENCE_COUNT = 3
# forward_seg()가 반환하는 2-class logit 중 어느 인덱스가 "fake"인지.
# AntiDeepfake 공식 evaluation.py 독스트링에 명시: "[Score] is [Fake logits, Real logits],
# [Label] is 0 for Groundtruth Fake and 1 for Groundtruth Real" - 즉 인덱스 0=fake, 1=real.
# (데스크탑 실측 검증: 실제 사람 음성이 FAKE_CLASS_INDEX=1일 때 score 0.99+로 나와 반전돼있음을 확인,
# 0으로 수정.)
FAKE_CLASS_INDEX = 0


def _safe_filename(filename: str) -> str:
    from pathlib import PureWindowsPath

    name = PureWindowsPath(filename).name
    return name if name and name not in (".", "..") else "upload"


def load_model(repo_dir: Path, checkpoint_path: Path) -> torch.nn.Module:
    sys.path.insert(0, str(repo_dir))
    from models.W2V import Model  # AntiDeepfake 저장소 코드 (repo_dir/models/W2V.py)
    from utils import load_weights  # AntiDeepfake 저장소 코드 (repo_dir/utils.py)

    model = Model(model_name="mms_300m")
    model.eval()
    state_dict = model.state_dict()
    # load_weights는 state_dict를 in-place로 채우는지 반환값을 써야 하는지 README/코드만으로는
    # 100% 확정하지 못했다 - 데스크탑에서 실제 체크포인트로 검증 필요.
    load_weights(state_dict, str(checkpoint_path))
    model.load_state_dict(state_dict)
    return model


def _decode_to_wav_if_needed(audio_path: Path) -> Path:
    """torchaudio(soundfile 백엔드)는 비압축 포맷(wav/flac/ogg)만 읽을 수 있다.
    폰 녹음은 보통 압축 포맷(m4a/mp4/mp3/aac)이라 실측(2026-08-25, 실제 카카오톡
    녹음 파일)에서 soundfile.LibsndfileError로 실패하는 걸 확인했다 - ffmpeg로 먼저
    wav로 변환한다. ffmpeg가 PATH에 있어야 한다(README 사전 준비 참고)."""
    if audio_path.suffix.lower() in {".wav", ".flac", ".ogg"}:
        return audio_path
    converted_path = audio_path.with_name(audio_path.stem + "_converted.wav")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), str(converted_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg으로 오디오 변환 실패({audio_path.suffix}): {result.stderr[-1000:]}"
        )
    return converted_path


def run_inference(model: torch.nn.Module, audio_path: Path) -> dict:
    audio_path = _decode_to_wav_if_needed(audio_path)
    waveform, sr = torchaudio.load(str(audio_path))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    with torch.no_grad():
        # forward_seg: 1번째 행 = 전체 오디오 예측, 나머지 행 = 프레임별 예측
        # (models/W2V.py Model.forward_seg 참고)
        seg_pred = model.forward_seg(waveform)

    probs = torch.softmax(seg_pred, dim=-1)
    overall_score = probs[0, FAKE_CLASS_INDEX].item()
    frame_scores = probs[1:, FAKE_CLASS_INDEX].tolist()

    return {"overall_score": overall_score, "frame_scores": frame_scores}


def build_evidence(frame_scores: list[float]) -> list[dict]:
    """프레임별 점수에서 임계값(0.5) 이상인 연속 구간을 병합해 근거 카드로 변환한다.
    구간을 점수 최댓값 기준 정렬해 상위 3개만 남긴다."""
    segments: list[tuple[int, int, float]] = []
    start = None
    peak = 0.0
    for i, score in enumerate(frame_scores):
        if score >= EVIDENCE_SCORE_THRESHOLD:
            if start is None:
                start = i
                peak = score
            else:
                peak = max(peak, score)
        elif start is not None:
            segments.append((start, i - 1, peak))
            start = None
    if start is not None:
        segments.append((start, len(frame_scores) - 1, peak))

    segments.sort(key=lambda s: s[2], reverse=True)
    top_segments = segments[:MAX_EVIDENCE_COUNT]

    evidence = []
    for start_idx, end_idx, _peak in top_segments:
        start_sec = round(start_idx * FRAME_STRIDE_SECONDS, 2)
        end_sec = round((end_idx + 1) * FRAME_STRIDE_SECONDS, 2)
        evidence.append(
            {
                "title": "시간 구간 이상 패턴",
                "description": f"{start_sec:.1f}초~{end_sec:.1f}초 구간에서 합성 흔적이 감지됨",
                "tags": ["temporal"],
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model = load_model(args.repo_dir, args.checkpoint)
    result = run_inference(model, args.input)

    score = result["overall_score"]
    evidence = build_evidence(result["frame_scores"]) if score >= EVIDENCE_SCORE_THRESHOLD else []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"score": score, "evidence": evidence}, ensure_ascii=False))


if __name__ == "__main__":
    main()
