import sys
from types import SimpleNamespace
import wave

import pytest
from fastapi import HTTPException

from app.audio_utils import clean_spoken_text
from app.tts import ChatterboxSynthesizer


class FakeChatterboxModel:
    def __init__(self):
        self.calls = []
        self.sr = 24000

    def generate(self, text, audio_prompt_path=None, **kwargs):
        self.calls.append(
            {
                "method": "generate",
                "text": text,
                "audio_prompt_path": audio_prompt_path,
                "kwargs": kwargs,
            }
        )
        return b"fake waveform"

    def generate_stream(self, text, audio_prompt_path=None, chunk_size=None, **kwargs):
        self.calls.append(
            {
                "method": "generate_stream",
                "text": text,
                "audio_prompt_path": audio_prompt_path,
                "chunk_size": chunk_size,
                "kwargs": kwargs,
            }
        )
        metrics = SimpleNamespace(latency_to_first_chunk=1.25)
        yield b"fake chunk 1", metrics
        yield b"fake chunk 2", metrics


def settings_for(tmp_path):
    return SimpleNamespace(
        tts_model="ResembleAI/chatterbox",
        tts_language="en",
        tts_device="cuda",
        tts_temperature=0.8,
        tts_top_p=0.95,
        tts_repetition_penalty=1.2,
        tts_norm_loudness=True,
        tts_chunk_size=25,
        tts_exaggeration=0.5,
        tts_cfg_weight=0.5,
        voice_id="default",
        voices_dir=tmp_path / "voices",
        audio_dir=tmp_path / "audio",
    )


def write_reference(tmp_path, voice_id, name, duration_seconds=6.0):
    voice_dir = tmp_path / "voices" / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    reference = voice_dir / name
    sample_rate = 22050
    frame_count = int(sample_rate * duration_seconds)
    with wave.open(str(reference), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\0\0" * frame_count)
    return reference


def test_clean_spoken_text_removes_emoji_markdown_and_urls():
    assert (
        clean_spoken_text("**Ready** to help \U0001f60a https://example.com")
        == "Ready to help"
    )


def test_clean_spoken_text_preserves_normal_speech():
    assert clean_spoken_text("I'm ready, and I can help.") == "I'm ready, and I can help."


def test_synthesize_rejects_non_speakable_text(tmp_path):
    write_reference(tmp_path, "default", "reference.wav")
    synthesizer = ChatterboxSynthesizer(settings_for(tmp_path))
    synthesizer._model = FakeChatterboxModel()

    with pytest.raises(HTTPException) as exc_info:
        synthesizer.synthesize("\U0001f60a", "default")

    assert exc_info.value.status_code == 400
    assert "speakable" in exc_info.value.detail


def test_short_synthesis_uses_one_generation_and_one_reference(tmp_path):
    expected_reference = write_reference(tmp_path, "default", "reference.wav")
    write_reference(tmp_path, "default", "reference-extra.wav")
    model = FakeChatterboxModel()
    saved_audio = []
    synthesizer = ChatterboxSynthesizer(settings_for(tmp_path))
    synthesizer._model = model
    synthesizer._save_audio = lambda path, wav, sample_rate: saved_audio.append((path, wav, sample_rate))

    result = synthesizer.synthesize("Hello there. Ready to help. \U0001f60a", "default")

    assert result.spoken_text == "Hello there. Ready to help."
    assert result.tts_input_chars == len("Hello there. Ready to help.")
    assert result.active_reference_count == 1
    assert len(model.calls) == 1
    assert model.calls[0]["text"] == "Hello there. Ready to help."
    assert model.calls[0]["audio_prompt_path"] == str(expected_reference)
    assert model.calls[0]["kwargs"]["temperature"] == 0.8
    assert model.calls[0]["kwargs"]["top_p"] == 0.95
    assert model.calls[0]["kwargs"]["repetition_penalty"] == 1.2
    assert model.calls[0]["kwargs"]["exaggeration"] == 0.5
    assert model.calls[0]["kwargs"]["cfg_weight"] == 0.5
    assert len(saved_audio) == 1
    assert saved_audio[0][1] == b"fake waveform"
    assert saved_audio[0][2] == 24000


def test_streaming_synthesis_yields_chunks_and_final_wav(tmp_path, monkeypatch):
    expected_reference = write_reference(tmp_path, "default", "reference.wav")
    model = FakeChatterboxModel()
    saved_audio = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cat=lambda chunks, dim: b"".join(chunks)),
    )
    synthesizer = ChatterboxSynthesizer(settings_for(tmp_path))
    synthesizer._model = model
    synthesizer._save_audio = lambda path, wav, sample_rate: saved_audio.append((path, wav, sample_rate))

    events = list(synthesizer.synthesize_stream("Hello stream.", "default"))

    assert len(events) == 3
    assert events[0].event["type"] == "chunk"
    assert events[0].event["chunk_index"] == 1
    assert events[0].event["first_latency_seconds"] == 1.25
    assert events[2].event["type"] == "final"
    assert events[2].event["total_chunks"] == 2
    assert events[2].event["audio_urls"] == [
        events[0].event["audio_url"],
        events[1].event["audio_url"],
    ]
    assert events[2].final_result.audio_url == events[2].event["audio_url"]
    assert model.calls[0]["method"] == "generate_stream"
    assert model.calls[0]["audio_prompt_path"] == str(expected_reference)
    assert model.calls[0]["chunk_size"] == 25
    assert len(saved_audio) == 3
    assert saved_audio[-1][1] == b"fake chunk 1fake chunk 2"


def test_synthesis_skips_short_reference_and_uses_longer_extra(tmp_path):
    write_reference(tmp_path, "default", "reference.wav", duration_seconds=2.0)
    expected_reference = write_reference(tmp_path, "default", "reference-extra.wav", duration_seconds=6.0)
    model = FakeChatterboxModel()
    synthesizer = ChatterboxSynthesizer(settings_for(tmp_path))
    synthesizer._model = model
    synthesizer._save_audio = lambda path, wav, sample_rate: None

    synthesizer.synthesize("Hello there.", "default")

    assert model.calls[0]["audio_prompt_path"] == str(expected_reference)


def test_synthesis_rejects_voice_when_all_references_are_too_short(tmp_path):
    write_reference(tmp_path, "default", "reference.wav", duration_seconds=2.0)
    synthesizer = ChatterboxSynthesizer(settings_for(tmp_path))
    synthesizer._model = FakeChatterboxModel()

    with pytest.raises(HTTPException) as exc_info:
        synthesizer.synthesize("Hello there.", "default")

    assert exc_info.value.status_code == 400
    assert "longer than 5 seconds" in exc_info.value.detail


def test_runtime_info_reports_chatterbox_provider(tmp_path):
    synthesizer = ChatterboxSynthesizer(settings_for(tmp_path))

    assert synthesizer.runtime_info()["provider"] == "chatterbox_streaming"
    assert synthesizer.runtime_info()["temperature"] == 0.8
    assert synthesizer.runtime_info()["chunk_size"] == 25
