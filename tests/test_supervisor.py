from pathlib import Path

from Scripts.supervisor import cached_snapshot_or_model_id


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
