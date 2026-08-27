import os
from functools import lru_cache
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        repo_dir = os.environ.get("SPAI_REPO_DIR")
        if not repo_dir:
            raise RuntimeError(
                "SPAI_REPO_DIR environment variable is required "
                "(absolute path to the cloned mever-team/spai repo)"
            )
        self.spai_repo_dir = Path(repo_dir)
        self.spai_python = os.environ.get("SPAI_PYTHON", "python")
        self.spai_cfg = os.environ.get("SPAI_CFG", "./configs/spai.yaml")
        self.spai_model = os.environ.get("SPAI_MODEL", "./weights/spai.pth")
        self.spai_timeout_seconds = int(os.environ.get("SPAI_TIMEOUT_SECONDS", "120"))
        # Full-resolution phone photos (3000px+) blow past an 8GB GPU's VRAM
        # in SPAI's patch-based forward pass, so downscale before inference.
        self.spai_resize_to = int(os.environ.get("SPAI_RESIZE_TO", "1024"))
        # Must be absolute: the SPAI subprocess runs with cwd=spai_repo_dir,
        # so a relative path here would resolve against the wrong directory.
        self.work_dir = Path(os.environ.get("SPAI_WORK_DIR", "./tmp")).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # --- AntiDeepfake(음성) 설정. SPAI와 마찬가지로 무거운 의존성(fairseq)은
        # 별도 conda env(antideepfake)에 격리하고, 여기(detection-api env)는 subprocess로만 부른다.
        # 이 클래스 하나에 두 모델 설정을 다 넣어서, SPAI_REPO_DIR/ANTIDEEPFAKE_REPO_DIR 둘 다
        # 설정돼야 get_settings()가 성공한다 - 데스크탑에는 어차피 둘 다 상시 구동되므로
        # 실질적 문제는 없지만, 한쪽만 쓰는 격리된 테스트 환경이라면 이 결합이 걸림돌이 될 수 있음(인지하고 있음).
        antideepfake_repo_dir = os.environ.get("ANTIDEEPFAKE_REPO_DIR")
        if not antideepfake_repo_dir:
            raise RuntimeError(
                "ANTIDEEPFAKE_REPO_DIR environment variable is required "
                "(absolute path to the cloned nii-yamagishilab/AntiDeepfake repo)"
            )
        self.antideepfake_repo_dir = Path(antideepfake_repo_dir)
        self.antideepfake_python = os.environ.get("ANTIDEEPFAKE_PYTHON", "python")
        self.antideepfake_script = Path(
            os.environ.get(
                "ANTIDEEPFAKE_SCRIPT",
                str(Path(__file__).resolve().parent.parent / "scripts" / "antideepfake_infer.py"),
            )
        )
        self.antideepfake_checkpoint = Path(
            os.environ.get("ANTIDEEPFAKE_CHECKPOINT", "./downloads/mms_300m.ckpt")
        )
        # 2.6분짜리 mp3가 실측 89초 걸림(2026-08-25) - 스펙 최대 길이(5분)까지 가면 120s를
        # 넘을 수 있어 300s로 늘림. Spring 쪽 detection.read-timeout도 같이 늘려야 함.
        self.antideepfake_timeout_seconds = int(os.environ.get("ANTIDEEPFAKE_TIMEOUT_SECONDS", "300"))
        self.antideepfake_work_dir = Path(os.environ.get("ANTIDEEPFAKE_WORK_DIR", "./tmp")).resolve()
        self.antideepfake_work_dir.mkdir(parents=True, exist_ok=True)

        # --- selimsef/dfdc_deepfake_challenge(영상, 얼굴조작 딥페이크) 설정. SPAI/AntiDeepfake와
        # 마찬가지로 무거운 의존성(opencv-python, facenet-pytorch 등)은 별도 conda env(dfdc)에
        # 격리하고, 여기(detection-api env)는 subprocess로만 부른다.
        dfdc_repo_dir = os.environ.get("DFDC_REPO_DIR")
        if not dfdc_repo_dir:
            raise RuntimeError(
                "DFDC_REPO_DIR environment variable is required "
                "(absolute path to the cloned selimsef/dfdc_deepfake_challenge repo)"
            )
        self.dfdc_repo_dir = Path(dfdc_repo_dir)
        self.dfdc_python = os.environ.get("DFDC_PYTHON", "python")
        self.dfdc_script = Path(
            os.environ.get(
                "DFDC_SCRIPT",
                str(Path(__file__).resolve().parent.parent / "scripts" / "dfdc_infer.py"),
            )
        )
        # selimsef 원본이 실제로 우승할 때 쓴 7개 체크포인트 풀 앙상블(predict_submission.sh/
        # download_weights.sh와 동일한 파일명) - 예전엔 "8GB GPU라 1개만" 이라는 이유로 하나만
        # 썼었는데, 그 GPU 메모리 근거가 실제로는 selimsef README에 없는 걸로 확인돼(README에
        # 있는 "12gb+"는 4-GPU 학습 요구사항이지 추론 요구사항이 아님) 2026-08-27 원복함.
        # DFDC_CHECKPOINTS는 쉼표로 구분한 경로 목록(기본값은 전부 ./weights/ 밑, 상대경로는
        # subprocess cwd=dfdc_repo_dir 기준으로 풀림 - dfdc_runner.py 참고).
        _default_dfdc_checkpoint_names = [
            "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
            "final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19",
            "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29",
            "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31",
            "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37",
            "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40",
            "final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23",
        ]
        default_dfdc_checkpoints = ",".join(f"./weights/{name}" for name in _default_dfdc_checkpoint_names)
        self.dfdc_checkpoints = [
            Path(p.strip())
            for p in os.environ.get("DFDC_CHECKPOINTS", default_dfdc_checkpoints).split(",")
        ]
        # 얼굴검출+CNN추론이 여러 프레임에 걸쳐 겹쳐 오디오(300s)보다 오래 걸릴 걸로 예상되나
        # 실측 데이터 없음(2026-08-26 기준) - 넉넉하게 잡고 실측 후 조정.
        self.dfdc_timeout_seconds = int(os.environ.get("DFDC_TIMEOUT_SECONDS", "600"))
        self.dfdc_work_dir = Path(os.environ.get("DFDC_WORK_DIR", "./tmp")).resolve()
        self.dfdc_work_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
