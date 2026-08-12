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


@lru_cache
def get_settings() -> Settings:
    return Settings()
