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
