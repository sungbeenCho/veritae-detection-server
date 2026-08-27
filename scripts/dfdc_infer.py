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
from __future__ import annotations  # dfdc conda env는 Python 3.9라 `str | None`(PEP 604) 같은
# 유니온 문법이 함수 정의 시점에 즉시 평가되면 TypeError가 남 - 이 import로 어노테이션 평가를
# 지연시켜야 함(2026-08-27, 데스크탑 첫 실행에서 실제로 발견된 버그. python=3.9로 만들도록
# README에 이미 명시돼 있던 제약을 코드 작성 시 확인 안 하고 넘어간 것이 원인).

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Windows에서 이 스크립트는 콘솔이 아니라 파이프(subprocess)로 실행되므로, stderr에 한글을
# print()하면 시스템 기본 코드페이지(한글 Windows면 cp949)로 인코딩된다. 받는 쪽
# (dfdc_runner.py)은 항상 UTF-8로 디코딩해서, cp949 바이트를 UTF-8로 잘못 해석해 한글이
# 깨진 채로 클라이언트까지 전달되는 문제가 있었다(2026-08-27, 데스크탑에서 실제로 발견 -
# "얼굴을 찾을 수 없습니다"가 "���� ..."로 깨져서 나옴). 코드페이지가
# 뭐든 상관없이 이 스크립트의 stderr 출력을 UTF-8로 고정한다.
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EVIDENCE_SCORE_THRESHOLD = 0.5
MAX_EVIDENCE_COUNT = 3
FRAMES_PER_VIDEO = 32
INPUT_SIZE = 380


class NoFaceDetectedError(Exception):
    """영상에서 얼굴을 하나도 못 찾은 경우 전용 예외. 일반 RuntimeError로 던지면 torch가
    CUDA 메모리 부족 등 진짜 서버 장애도 RuntimeError로 던지는 경우와 구분이 안 돼서,
    "정상적인 사용자 케이스(얼굴 없는 영상)"와 "진짜 오류"를 dfdc_runner.py/video.py가
    각각 다른 방식으로 처리할 수 있도록 전용 타입으로 분리했다(2026-08-27)."""
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
        raise NoFaceDetectedError("얼굴을 찾을 수 없습니다.")

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
    target_layer(model.encoder.conv_head)는 timm 소스(efficientnet.py) 직접 확인 완료 -
    forward_features() 안에서 실제로 호출되는 실제 속성이 맞다(2026-08-27 재검증).

    실제 원인이었던 버그(2026-08-27, 데스크탑 실측에서 evidenceImage가 계속 null로만 나와서
    발견): model은 load_models()에서 .half()(fp16)로 만들어지는데, 여기 input_tensor는
    .float()(fp32)로 넘어가 dtype 불일치 RuntimeError가 나고 위 except가 조용히 삼켜버렸다.
    model.float()로 맞춰준다 - 이 함수 호출 이후 main()에서 이 model을 다시 안 쓰므로
    안전하게 in-place로 바꿀 수 있다.

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

        model.float()
        target_layer = model.encoder.conv_head
        cam = GradCAM(model=model, target_layers=[target_layer])
        input_tensor = face_tensor.unsqueeze(0).float()
        # 2026-08-27: 설치된 pytorch_grad_cam 버전은 __call__(input_tensor, targets, ...)의
        # targets에 기본값이 없어(base_cam.py 확인) targets 생략 시 TypeError가 났다.
        # targets=None을 명시하면 내부 forward()가 출력값 argmax로 자동 타깃을 잡는데,
        # 이 모델은 출력이 1개(이진 sigmoid)뿐이라 그 하나가 그대로 타깃이 된다.
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

        rgb_face = face_preprocessed_rgb.astype(np.float32) / 255.0
        rgb_face = cv2.resize(rgb_face, (grayscale_cam.shape[1], grayscale_cam.shape[0]))
        overlay = show_cam_on_image(rgb_face, grayscale_cam, use_rgb=True)

        success, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not success:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        # 2026-08-27: dtype 수정 이후에도 계속 null이 나와서, 원인을 숨기지 않고 실제
        # traceback을 stderr에 남기도록 임시로 바꿈(진짜 원인 확인 전까지 유지 - 원인
        # 확정되면 다시 조용히 삼키는 형태로 되돌릴 것). 이 print는 stderr로만 가서
        # main()의 최종 리턴값(JSON)에는 안 섞인다.
        import traceback
        traceback.print_exc(file=sys.stderr)
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
    except NoFaceDetectedError as e:
        # 얼굴 없음은 "정상적인 사용자 케이스"라 종료 코드를 2로 따로 둬서, dfdc_runner.py가
        # 진짜 오류(exit 1)와 구분해 다르게 처리할 수 있게 한다(2026-08-27).
        print(str(e), file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        # 그 외 RuntimeError(예: torch의 CUDA 메모리 부족 등 진짜 오류)는 여기서 잡아서
        # 메시지만 stderr에 출력한다. 안 잡으면 파이썬이 전체 traceback을 stderr에 쏟아내고,
        # dfdc_runner.py가 stderr의 마지막 2000자만 잘라 전달하는 과정에서 정작 중요한
        # 메시지(traceback 앞부분)가 잘려나갈 위험이 있다.
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
