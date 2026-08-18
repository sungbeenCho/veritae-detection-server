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
        self.antideepfake_timeout_seconds = int(os.environ.get("ANTIDEEPFAKE_TIMEOUT_SECONDS", "120"))
        self.antideepfake_work_dir = Path(os.environ.get("ANTIDEEPFAKE_WORK_DIR", "./tmp")).resolve()
        self.antideepfake_work_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
