import importlib


def test_default_model_is_gemma4(monkeypatch, tmp_path):
    monkeypatch.delenv("ROBIT_LLM_MODEL", raising=False)
    monkeypatch.setenv("ROBIT_DATA_DIR", str(tmp_path))

    config = importlib.import_module("app.config")
    settings = config.load_settings()

    assert settings.llm_model == "gemma4:e4b"
    assert settings.llm_provider == "openai_compatible"
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_think is False
    assert settings.realtime_ws_url == "ws://localhost:7861/v1/realtime"
    assert settings.realtime_voice == "serena"
    assert settings.vision_enabled is True
    assert settings.vision_base_url == "http://127.0.0.1:8081/v1"
    assert settings.vision_model == "ggml-org/gemma-4-E4B-it-GGUF:Q4_0"
    assert settings.vision_image_tokens == 140
    assert settings.vision_awareness_interval_seconds == 5.0
    assert settings.camera_frame_interval_seconds == 5.0
    assert settings.vision_snapshot_ttl_seconds == 10.0


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBIT_LLM_MODEL", "custom:model")
    monkeypatch.setenv("ROBIT_LLM_THINK", "true")
    monkeypatch.setenv("ROBIT_REALTIME_WS_URL", "ws://127.0.0.1:9000/v1/realtime")
    monkeypatch.setenv("ROBIT_REALTIME_VOICE", "serena")
    monkeypatch.setenv("ROBIT_REALTIME_MODEL", "repo/e4b:Q8_0")
    monkeypatch.delenv("ROBIT_VISION_MODEL", raising=False)
    monkeypatch.setenv("ROBIT_VISION_BASE_URL", "http://127.0.0.1:9999/v1/")
    monkeypatch.setenv("ROBIT_DATA_DIR", str(tmp_path))

    config = importlib.import_module("app.config")
    settings = config.load_settings()

    assert settings.llm_model == "custom:model"
    assert settings.llm_think is True
    assert settings.realtime_ws_url == "ws://127.0.0.1:9000/v1/realtime"
    assert settings.realtime_voice == "serena"
    assert settings.vision_model == "repo/e4b:Q8_0"
    assert settings.vision_base_url == "http://127.0.0.1:9999/v1"
