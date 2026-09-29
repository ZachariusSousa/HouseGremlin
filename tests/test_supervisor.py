from pathlib import Path

from Scripts.supervisor import Service, cached_snapshot_or_model_id, open_dashboard, order_services_for_startup


def test_cached_snapshot_or_model_id_uses_local_main_snapshot(tmp_path: Path) -> None:
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    snapshot = (
        tmp_path
        / "models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice"
        / "snapshots"
        / "revision-123"
    )
    snapshot.mkdir(parents=True)
    refs = snapshot.parents[1] / "refs"
    refs.mkdir()
    (refs / "main").write_text("revision-123\n", encoding="utf-8")

    assert cached_snapshot_or_model_id(model_id, tmp_path) == str(snapshot)


def test_cached_snapshot_or_model_id_keeps_model_id_when_not_cached(tmp_path: Path) -> None:
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    assert cached_snapshot_or_model_id(model_id, tmp_path) == model_id


def _service(name: str) -> Service:
    return Service(name, [], Path.cwd(), {}, lambda: True, 1)


def test_startup_order_starts_the_frontend_before_slow_sidecars() -> None:
    services = [_service("llama-server"), _service("voice"), _service("pc-brain"), _service("tracking")]

    assert [service.name for service in order_services_for_startup(services)] == [
        "pc-brain",
        "llama-server",
        "voice",
        "tracking",
    ]


def test_open_dashboard_uses_the_configured_local_port() -> None:
    opened: list[str] = []

    open_dashboard(8088, opened.append)

    assert opened == ["http://localhost:8088"]
