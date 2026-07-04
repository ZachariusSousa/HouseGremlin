import importlib


def test_default_model_is_gemma4(monkeypatch, tmp_path):
    monkeypatch.delenv("ROBIT_LLM_MODEL", raising=False)
    monkeypatch.setenv("ROBIT_DATA_DIR", str(tmp_path))

    config = importlib.import_module("app.config")
    settings = config.load_settings()

    assert settings.llm_model == "gemma4:e4b"
    assert settings.llm_think is False


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBIT_LLM_MODEL", "custom:model")
    monkeypatch.setenv("ROBIT_LLM_THINK", "true")
    monkeypatch.setenv("ROBIT_DATA_DIR", str(tmp_path))

    config = importlib.import_module("app.config")
    settings = config.load_settings()

    assert settings.llm_model == "custom:model"
    assert settings.llm_think is True
