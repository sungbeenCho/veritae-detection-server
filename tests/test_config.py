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


def test_gpu_queue_settings_have_sane_defaults(monkeypatch):
    monkeypatch.setenv("SPAI_REPO_DIR", "C:/fake/spai")
    monkeypatch.setenv("ANTIDEEPFAKE_REPO_DIR", "C:/fake/antideepfake")
    monkeypatch.setenv("DFDC_REPO_DIR", "C:/fake/dfdc")
    monkeypatch.delenv("GPU_QUEUE_MAX_CONCURRENT", raising=False)
    monkeypatch.delenv("GPU_QUEUE_MAX_DEPTH", raising=False)

    settings = Settings()

    assert settings.gpu_queue_max_concurrent == 1
    assert settings.gpu_queue_max_depth == 10


def _set_required_repo_dirs(monkeypatch):
    monkeypatch.setenv("SPAI_REPO_DIR", "C:/fake/spai")
    monkeypatch.setenv("ANTIDEEPFAKE_REPO_DIR", "C:/fake/antideepfake")
    monkeypatch.setenv("DFDC_REPO_DIR", "C:/fake/dfdc")


def test_misinfo_nli_model_defaults_to_augmented_model(monkeypatch):
    """서버 기본 NLI는 보충 학습한 모델(data/nli_model_aug)이어야 한다. 옛 모델(data/nli_model)은 "~라는 이야기를
    들었어요" 같은 전해 들은 말의 정답 근거에 0.058점을 줘서 반박을 버렸다(2026-10-13 실측 기록 비교)."""
    _set_required_repo_dirs(monkeypatch)
    monkeypatch.delenv("MISINFO_NLI_MODEL", raising=False)

    path = Settings().misinfo_nli_model

    assert path.name == "nli_model_aug"
    assert path.parent.name == "data"


def test_misinfo_nli_model_env_override_is_respected(monkeypatch):
    _set_required_repo_dirs(monkeypatch)
    monkeypatch.setenv("MISINFO_NLI_MODEL", "D:/models/custom_nli")

    assert Settings().misinfo_nli_model == Path("D:/models/custom_nli")


def test_misinfo_nli_model_default_is_same_for_server_and_regression_script(monkeypatch):
    """회귀 테스트(nli_check.default_model_path)와 실서버(Settings)의 기본 모델이 어긋나면, 테스트는 통과하는데
    서버는 다른 모델로 도는 일이 생긴다(2026-10-13: 테스트 21/30, 서버는 옛 모델이라 탐지 실패)."""
    from nli_check import default_model_path

    _set_required_repo_dirs(monkeypatch)
    monkeypatch.delenv("MISINFO_NLI_MODEL", raising=False)

    assert Path(default_model_path()) == Settings().misinfo_nli_model


def test_misinfo_wiki_search_is_off_unless_explicitly_enabled(monkeypatch):
    """위키백과 검색은 사용자 문장의 핵심 단어를 외부로 보내므로 기본은 꺼짐이어야 한다(2026-10-01)."""
    monkeypatch.setenv("SPAI_REPO_DIR", "C:/fake/spai")
    monkeypatch.setenv("ANTIDEEPFAKE_REPO_DIR", "C:/fake/antideepfake")
    monkeypatch.setenv("DFDC_REPO_DIR", "C:/fake/dfdc")
    monkeypatch.delenv("MISINFO_WIKI_SEARCH", raising=False)
    assert Settings().misinfo_wiki_search is False

    monkeypatch.setenv("MISINFO_WIKI_SEARCH", "1")
    assert Settings().misinfo_wiki_search is True
