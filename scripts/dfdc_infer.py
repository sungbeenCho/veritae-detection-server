#!/usr/bin/env python
"""dfdc_infer.py

selimsef/dfdc_deepfake_challenge(단일 체크포인트, tf_efficientnet_b7_ns)로 영상 파일 하나를
추론해서, 전체 score와 시간 구간별(frame-level) evidence, (best-effort) 얼굴 히트맵
이미지를 JSON으로 저장한다.

`dfdc` conda env(opencv-python, facenet-pytorch, pytorch-grad-cam 등 설치됨)에서
실행되어야 한다. veritae-detection-server(FastAPI, detection-api env)는 이 스크립트를
subprocess로 호출하고 --output 경로의 JSON만 읽는다 - SPAI/AntiDeepfake 연동과 동일한 패턴.

selimsef 저장소는 프레임 추출에 ffmpeg가 아니라 OpenCV(cv2.VideoCapture)를 쓴다
(kernel_utils.VideoReader) - 얼굴 검출은 facenet-pytorch의 MTCNN
(kernel_utils.FaceExtractor). 이 스크립트는 그 두 클래스를 그대로 재사용한다.

원본 predict_on_video()(kernel_utils.py)는 프레임별 점수를 평균 내서(np.mean) 값 하나만
반환하고 프레임별 점수는 버린다 - 시간 구간 evidence를 만들려면 평균 내기 전 값이 필요해서
그 로직을 그대로 못 쓰고 이 파일에서 재구현했다
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


def _safe_filename(filename: str) -> str:
    from pathlib import PureWindowsPath

    name = PureWindowsPath(filename).name
    return name if name and name not in (".", "..") else "upload"


def load_model(repo_dir: Path, checkpoint_path: Path):
    sys.path.insert(0, str(repo_dir))
    from training.zoo.classifiers import DeepFakeClassifier  # selimsef 저장소 코드

    model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
    model.eval()
    return model.half()


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else DEFAULT_FPS


def run_inference(model, repo_dir: Path, video_path: Path) -> dict:
    """selimsef 저장소의 VideoReader/FaceExtractor를 그대로 재사용해 얼굴 프레임을 뽑고,
    predict_on_video()가 버리는 프레임별 점수(평균 내기 전 값)를 직접 계산한다."""
    sys.path.insert(0, str(repo_dir))
    from kernel_utils import FaceExtractor, VideoReader, isotropically_resize_image, normalize_transform, put_to_center

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
        y_pred = model(x_tensor.half())
        frame_scores = torch.sigmoid(y_pred.squeeze()).float().cpu().numpy()
    if frame_scores.ndim == 0:
        frame_scores = np.array([frame_scores.item()])

    overall_score = float(np.mean(frame_scores))
    worst_idx = int(np.argmax(frame_scores))

    return {
        "overall_score": overall_score,
        "frame_idxs": frame_idxs,
        "frame_scores": frame_scores.tolist(),
        "fps": fps,
        "worst_frame_bgr": faces_by_frame_idx[worst_idx][1],
        "worst_frame_tensor": x_tensor[worst_idx],
        "model_ref": model,
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


def try_generate_heatmap(model, face_tensor, face_bgr) -> str | None:
    """가장 의심스러운 프레임에 Grad-CAM을 적용해 base64 PNG를 반환한다. 실패하면 조용히
    None을 반환한다(전체 분석 실패로 이어지면 안 되므로 여기서 예외를 삼킨다) - 설계 §4 fallback.
    target_layer(model.encoder.conv_head)는 timm의 EfficientNet 구조를 근거로 추정한 것으로,
    실제 클래스 속성명이 다르면 데스크탑에서 실제 model 객체를 찍어보고 수정해야 한다."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        target_layer = model.encoder.conv_head
        cam = GradCAM(model=model, target_layers=[target_layer])
        input_tensor = face_tensor.unsqueeze(0).float()
        grayscale_cam = cam(input_tensor=input_tensor)[0]

        rgb_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
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
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model = load_model(args.repo_dir, args.checkpoint)
    result = run_inference(model, args.repo_dir, args.input)

    score = result["overall_score"]
    evidence = []
    evidence_image = None
    if score >= EVIDENCE_SCORE_THRESHOLD:
        evidence = build_evidence(result["frame_idxs"], result["frame_scores"], result["fps"])
        evidence_image = try_generate_heatmap(
            result["model_ref"], result["worst_frame_tensor"], result["worst_frame_bgr"]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"score": score, "evidence": evidence, "evidence_image": evidence_image}, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
