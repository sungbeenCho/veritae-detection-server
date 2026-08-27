from pathlib import Path

from app.config import Settings


def test_antideepfake_script_default_points_to_existing_file(monkeypatch):
    # SPAI_REPO_DIR/ANTIDEEPFAKE_REPO_DIR/DFDC_REPO_DIR are required for Settings() to
    # construct but their actual values don't matter for this test - only
    # ANTIDEEPFAKE_SCRIPT's computed default does. Values below need not exist on disk.
    monkeypatch.setenv("SPAI_REPO_DIR", "C:/fake/spai")
    monkeypatch.setenv("ANTIDEEPFAKE_REPO_DIR", "C:/fake/antideepfake")
    monkeypatch.setenv("DFDC_REPO_DIR", "C:/fake/dfdc")
    monkeypatch.delenv("ANTIDEEPFAKE_SCRIPT", raising=False)

    settings = Settings()

    # Regression test for an off-by-one `.parent` bug: the computed default must
    # resolve to a real file at <repo-root>/scripts/antideepfake_infer.py, not to
    # some directory above the repo root.
    assert settings.antideepfake_script.is_file()
    assert settings.antideepfake_script.name == "antideepfake_infer.py"


def test_dfdc_checkpoints_default_is_full_7_checkpoint_ensemble(monkeypatch):
    # Regression test: 예전에 GPU 메모리를 이유로 체크포인트 1개로 줄였던 게 근거 없는
    # 판단으로 밝혀져 selimsef 원본(predict_submission.sh/download_weights.sh)과 동일한
    # 7개 풀 앙상블로 원복했다(2026-08-27) - 다시 실수로 1개로 줄지 않게 개수/파일명을 고정.
    monkeypatch.setenv("SPAI_REPO_DIR", "C:/fake/spai")
    monkeypatch.setenv("ANTIDEEPFAKE_REPO_DIR", "C:/fake/antideepfake")
    monkeypatch.setenv("DFDC_REPO_DIR", "C:/fake/dfdc")
    monkeypatch.delenv("DFDC_CHECKPOINTS", raising=False)

    settings = Settings()

    names = [p.name for p in settings.dfdc_checkpoints]
    assert names == [
        "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
        "final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19",
        "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29",
        "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31",
        "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37",
        "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40",
        "final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23",
    ]


def test_dfdc_checkpoints_env_override_splits_on_comma(monkeypatch):
    monkeypatch.setenv("SPAI_REPO_DIR", "C:/fake/spai")
    monkeypatch.setenv("ANTIDEEPFAKE_REPO_DIR", "C:/fake/antideepfake")
    monkeypatch.setenv("DFDC_REPO_DIR", "C:/fake/dfdc")
    monkeypatch.setenv("DFDC_CHECKPOINTS", "./weights/a, ./weights/b")

    settings = Settings()

    assert settings.dfdc_checkpoints == [Path("./weights/a"), Path("./weights/b")]
