#!/usr/bin/env python
"""dfdc_infer.py

selimsef/dfdc_deepfake_challenge(7개 체크포인트 풀 앙상블, tf_efficientnet_b7_ns)로 영상
파일 하나를 추론해서, 전체 score와 시간 구간별(frame-level) evidence, (best-effort) 얼굴
히트맵 이미지를 JSON으로 저장한다. 앙상블 구성은 predict_submission.sh/download_weights.sh와
동일 - kernel_utils.predict_on_video()의 방식(모델별로 먼저 confident_strategy 집계 후 그
집계값들을 모델 간 단순 평균)을 그대로 재현한다(2026-08-27, 예전엔 GPU 메모리를 이유로
1개만 썼었는데 그 근거가 실제로는 잘못된 것으로 확인돼 원복 - config.py 주석 참고).

`dfdc` conda env(opencv-python, facenet-pytorch, grad-cam 등 설치됨)에서
실행되어야 한다. veritae-detection-server(FastAPI, detection-api env)는 이 스크립트를
subprocess로 호출하고 --output 경로의 JSON만 읽는다 - SPAI/AntiDeepfake 연동과 동일한 패턴.

selimsef 저장소는 프레임 추출에 ffmpeg가 아니라 OpenCV(cv2.VideoCapture)를 쓴다
(kernel_utils.VideoReader) - 얼굴 검출은 facenet-pytorch의 MTCNN
(kernel_utils.FaceExtractor). 이 스크립트는 그 두 클래스를 그대로 재사용한다.

원본 predict_on_video()(kernel_utils.py)는 프레임별 점수를 strategy 함수(predict_folder.py가
넘기는 confident_strategy)로 집계한 값 하나만 반환하고 프레임별 점수는 버린다 - 시간 구간
evidence를 만들려면 집계 전 값이 필요해서 그 로직을 그대로 못 쓰고 이 파일에서 재구현했다.
최종 overall_score는 원본과 동일하게 confident_strategy로 집계한다(단순 np.mean이 아님)
(docs(veritae-server 레포): docs/superpowers/specs/2026-08-26-video-ai-detection-design.md §3, §4).

⚠️ Grad-CAM(공간적 근거) 부분은 아직 데스크탑에서 검증 안 됨 - selimsef가 순수 CNN
(EfficientNet)이라 적용 가능성은 높다고 판단했지만 실제로 말이 되는 히트맵이 나오는지는
미확인이다. best-effort로 시도하고 실패하면 evidence_image를 null로 둔다(설계 §4 fallback).
"""
import argparse
import base64
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

EVIDENCE_SCORE_THRESHOLD = 0.5
MAX_EVIDENCE_COUNT = 3
FRAMES_PER_VIDEO = 32
INPUT_SIZE = 380
DEFAULT_FPS = 30.0  # cv2가 fps를 못 읽을 때의 추정치 - 실측 필요
# 검출된 얼굴 수만큼 배치 텐서를 무제한 할당하면(np.zeros((len(faces), ...)))
# 다인물 영상에서 8GB GPU가 OOM 날 수 있다. selimsef 원본(kernel_utils.predict_on_video)도
# batch_size*4개로 상한을 두고 그 이상은 조용히 버리는데, 여기서도 같은 방어를 최소 형태로 둔다
# (인물별 분리 등 정교한 처리는 이번 스코프 밖).
MAX_FACES = 32


def _safe_filename(filename: str) -> str:
    from pathlib import PureWindowsPath

    name = PureWindowsPath(filename).name
    return name if name and name not in (".", "..") else "upload"


def load_models(repo_dir: Path, checkpoint_paths: list[Path]) -> list:
    sys.path.insert(0, str(repo_dir))
    from training.zoo.classifiers import DeepFakeClassifier  # selimsef 저장소 코드

    models = []
    for checkpoint_path in checkpoint_paths:
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
        model.eval()
        models.append(model.half())
    return models


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else DEFAULT_FPS


def run_inference(models: list, repo_dir: Path, video_path: Path) -> dict:
    """selimsef 저장소의 VideoReader/FaceExtractor를 그대로 재사용해 얼굴 프레임을 뽑고,
    predict_on_video()가 버리는 프레임별 점수(평균 내기 전 값)를 직접 계산한다. models는
    앙상블 구성 체크포인트 전부(보통 7개) - 얼굴 검출/전처리는 한 번만 하고 같은 입력
    텐서를 모델마다 돌린다."""
    sys.path.insert(0, str(repo_dir))
    from kernel_utils import (
        FaceExtractor,
        VideoReader,
        confident_strategy,
        isotropically_resize_image,
        normalize_transform,
        put_to_center,
    )

    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=FRAMES_PER_VIDEO)
    face_extractor = FaceExtractor(video_read_fn)

    frame_data_list = face_extractor.process_video(str(video_path))

    faces_by_frame_idx = []
    for frame_data in frame_data_list:
        for face in frame_data["faces"]:
            faces_by_frame_idx.append((frame_data["frame_idx"], face))

    if not faces_by_frame_idx:
        raise RuntimeError("얼굴을 찾을 수 없습니다.")

    # 다인물 영상 등 얼굴이 과도하게 많이 검출된 경우 배치 크기가 무제한으로 커지는 걸 막는다.
    # 같은 frame_idx에 여러 얼굴이 잡혀도 그대로 병합되므로(인물 분리는 하지 않음) 정교하진
    # 않지만, 최소한 OOM은 방지한다.
    faces_by_frame_idx = faces_by_frame_idx[:MAX_FACES]

    fps = get_fps(video_path)
    frame_idxs = [idx for idx, _ in faces_by_frame_idx]

    x = np.zeros((len(faces_by_frame_idx), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    for i, (_, face) in enumerate(faces_by_frame_idx):
        resized_face = isotropically_resize_image(face, INPUT_SIZE)
        x[i] = put_to_center(resized_face, INPUT_SIZE)

    x_tensor = torch.tensor(x, device="cuda").float()
    x_tensor = x_tensor.permute((0, 3, 1, 2))
    for i in range(len(x_tensor)):
        x_tensor[i] = normalize_transform(x_tensor[i] / 255.0)

    with torch.no_grad():
        per_model_frame_scores = []
        for model in models:
            y_pred = model(x_tensor.half())
            scores = torch.sigmoid(y_pred.squeeze()).float().cpu().numpy()
            if scores.ndim == 0:
                scores = np.array([scores.item()])
            per_model_frame_scores.append(scores)

    # selimsef 원본(kernel_utils.predict_on_video)의 앙상블 방식을 그대로 재현: 프레임별
    # 점수를 단순 평균하지 않고, 모델마다 먼저 confident_strategy로 집계한 뒤(고신뢰 fake
    # 프레임이 충분히 많으면 그 프레임들만 평균, 거의 전부 real이면 그쪽만 평균, 아니면 전체
    # 평균) 그 7개 집계값을 모델 간 단순 평균한다(np.mean(preds), kernel_utils.py 참고).
    per_model_overall_scores = [confident_strategy(scores) for scores in per_model_frame_scores]
    overall_score = float(np.mean(per_model_overall_scores))

    # 시간 구간 evidence는 selimsef 원본에 없는 우리 자체 기능이라 원본이 참고할 방법이
    # 없다 - 프레임별로 7개 모델 점수를 평균한 곡선을 구간 추출에 쓴다(overall_score 산식과
    # 완전히 같지는 않지만(위는 모델별 집계 후 평균, 이건 프레임별 평균), "어느 구간이
    # 의심스러운가"를 보여주는 목적에는 프레임 단위 정보가 필요해 이쪽이 더 맞다).
    frame_scores = np.mean(per_model_frame_scores, axis=0)
    worst_idx = int(np.argmax(frame_scores))

    return {
        "overall_score": overall_score,
        "frame_idxs": frame_idxs,
        "frame_scores": frame_scores.tolist(),
        "fps": fps,
        # CAM은 전처리된(레터박스 padding 포함) 380x380 텐서로 계산되므로, 히트맵 배경도
        # 전처리 전 원본 크롭이 아니라 이 배열(x[worst_idx])을 써야 좌표계가 일치한다.
        "worst_frame_preprocessed": x[worst_idx],
        "worst_frame_tensor": x_tensor[worst_idx],
        # Grad-CAM을 7개 모델 전부에 대해 계산해 합치는 표준적인 방법이 없고(모델마다
        # activation 스케일이 달라 CAM을 평균/합성하는 게 의미적으로 애매함), best-effort
        # 기능에 그 정도 복잡도를 들일 이유가 약해 첫 번째 체크포인트로만 계산한다 -
        # 실패해도 시간 구간 evidence(모델 7개 평균 기반)는 항상 나온다.
        "model_ref": models[0],
    }


def build_evidence(frame_idxs: list[int], frame_scores: list[float], fps: float) -> list[dict]:
    """프레임별 점수에서 임계값(0.5) 이상인 연속 구간을 병합해 근거 카드로 변환한다.
    구간을 점수 최댓값 기준 정렬해 상위 3개만 남긴다 (antideepfake_infer.py의
    build_evidence와 동일한 로직 - frame_idx -> 초 변환만 fps 기반으로 다름, AntiDeepfake는
    고정 프레임 간격(20ms)을 쓰지만 영상은 프레임 인덱스가 균등 샘플링돼 fps가 필요함)."""
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
    for start_i, end_i, _peak in top_segments:
        start_sec = round(frame_idxs[start_i] / fps, 2)
        end_sec = round(frame_idxs[end_i] / fps, 2)
        evidence.append(
            {
                "title": "얼굴 조작 의심 구간",
                "description": f"{start_sec:.1f}초~{end_sec:.1f}초 구간에서 얼굴 합성 흔적이 감지됨",
                "tags": ["temporal", "face-swap"],
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )
    return evidence


def try_generate_heatmap(model, face_tensor, face_preprocessed_rgb) -> str | None:
    """가장 의심스러운 프레임에 Grad-CAM을 적용해 base64 PNG를 반환한다. 실패하면 조용히
    None을 반환한다(전체 분석 실패로 이어지면 안 되므로 여기서 예외를 삼킨다) - 설계 §4 fallback.
    target_layer(model.encoder.conv_head)는 timm의 EfficientNet 구조를 근거로 추정한 것으로,
    실제 클래스 속성명이 다르면 데스크탑에서 실제 model 객체를 찍어보고 수정해야 한다.

    face_preprocessed_rgb는 run_inference가 만든 전처리된 배열(x[worst_idx], INPUT_SIZE 크기로
    isotropically_resize_image+put_to_center 레터박스 처리됨)이어야 한다 - CAM(face_tensor에서
    계산됨)과 좌표계가 정확히 일치하는 배경 이미지는 이것뿐이다. 원본 크롭(전처리 전, 크기도 다름)을
    쓰면 히트맵이 얼굴의 엉뚱한 위치를 가리킬 수 있다.

    selimsef의 VideoReader._postprocess_frame()이 프레임을 읽을 때 이미 cv2.COLOR_BGR2RGB를
    적용한다(kernel_utils.py 확인됨) - 즉 face_preprocessed_rgb는 이미 RGB다. 여기서 다시
    BGR2RGB 변환을 하면 R/B 채널이 재반전되어 히트맵 배경이 파랗게 뜨므로 변환하지 않는다."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        target_layer = model.encoder.conv_head
        cam = GradCAM(model=model, target_layers=[target_layer])
        input_tensor = face_tensor.unsqueeze(0).float()
        grayscale_cam = cam(input_tensor=input_tensor)[0]

        rgb_face = face_preprocessed_rgb.astype(np.float32) / 255.0
        rgb_face = cv2.resize(rgb_face, (grayscale_cam.shape[1], grayscale_cam.shape[0]))
        overlay = show_cam_on_image(rgb_face, grayscale_cam, use_rgb=True)

        success, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not success:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path, nargs="+")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        models = load_models(args.repo_dir, args.checkpoints)
        result = run_inference(models, args.repo_dir, args.input)
    except RuntimeError as e:
        # run_inference가 얼굴 미검출 시 던지는 RuntimeError(한국어 메시지)는 여기서 잡아서
        # 메시지만 stderr에 출력한다. 안 잡으면 파이썬이 전체 traceback을 stderr에 쏟아내고,
        # dfdc_runner.py가 stderr의 마지막 2000자만 잘라 전달하는 과정에서 정작 중요한
        # 한국어 메시지(traceback 앞부분)가 잘려나갈 위험이 있다.
        print(str(e), file=sys.stderr)
        sys.exit(1)

    score = result["overall_score"]
    evidence = []
    evidence_image = None
    if score >= EVIDENCE_SCORE_THRESHOLD:
        evidence = build_evidence(result["frame_idxs"], result["frame_scores"], result["fps"])
        evidence_image = try_generate_heatmap(
            result["model_ref"], result["worst_frame_tensor"], result["worst_frame_preprocessed"]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"score": score, "evidence": evidence, "evidence_image": evidence_image}, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
