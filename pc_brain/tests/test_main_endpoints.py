import json
from dataclasses import dataclass
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import main
from app.tts import SynthesisStreamEvent
from app.voices import VoiceStore


@dataclass
class FakeChatResult:
    response: str
    model: str


@dataclass
class FakeTranscriptionResult:
    text: str
    language: str | None = "en"
    duration_seconds: float | None = 1.0


@dataclass
class FakeSynthesisResult:
    audio_url: str
    audio_urls: list[str]
    voice_id: str
    spoken_text: str = "reply to hello"
    tts_input_chars: int = 14
    active_reference_count: int = 1


class FakeLlmClient:
    async def chat(self, text: str):
        return FakeChatResult(response=f"reply to {text}", model="gemma4:e4b")


class FakeTranscriber:
    def transcribe(self, path):
        return FakeTranscriptionResult(text="hello robit")


class FakeTts:
    def runtime_info(self):
        return {"provider": "chatterbox_streaming", "model_loaded": False}

    def synthesize(self, text: str, voice_id: str):
        return FakeSynthesisResult(
            audio_url="/audio/fake.wav",
            audio_urls=["/audio/fake.wav"],
            voice_id=voice_id,
            spoken_text=text,
            tts_input_chars=len(text),
        )

    def synthesize_stream(self, text: str, voice_id: str):
        yield SynthesisStreamEvent(
            event={
                "type": "chunk",
                "chunk_index": 1,
                "audio_url": "/audio/chunk.wav",
                "voice_id": voice_id,
                "first_latency_seconds": 1.25,
            }
        )
        yield SynthesisStreamEvent(
            event={
                "type": "final",
                "audio_url": "/audio/final.wav",
                "audio_urls": ["/audio/chunk.wav"],
                "voice_id": voice_id,
                "spoken_text": text,
                "tts_input_chars": len(text),
                "active_reference_count": 1,
                "total_chunks": 1,
            },
            final_result=FakeSynthesisResult(
                audio_url="/audio/final.wav",
                audio_urls=["/audio/chunk.wav"],
                voice_id=voice_id,
                spoken_text=text,
                tts_input_chars=len(text),
            ),
        )


class FakeVoiceStore:
    def list_voice_ids(self):
        return ["default", "narrator"]

    def sample_count(self, voice_id: str):
        return {"default": 1, "narrator": 2}[voice_id]


def test_voices_endpoint_lists_available_voices(monkeypatch):
    monkeypatch.setattr(main, "voice_store", FakeVoiceStore())

    response = TestClient(main.app).get("/voices")

    assert response.status_code == 200
    assert response.json()["voices"] == ["default", "narrator"]
    assert response.json()["voice_details"] == [
        {"voice_id": "default", "sample_count": 1},
        {"voice_id": "narrator", "sample_count": 2},
    ]
    assert response.json()["default_voice_id"] == "default"


def test_voice_store_counts_multiple_reference_clips(tmp_path):
    voice_dir = tmp_path / "voices" / "narrator"
    voice_dir.mkdir(parents=True)
    (voice_dir / "reference.wav").write_bytes(b"fake")
    (voice_dir / "reference-extra.wav").write_bytes(b"fake")
    (voice_dir / "notes.txt").write_text("ignored")

    store = VoiceStore(SimpleNamespace(voices_dir=tmp_path / "voices"))

    assert store.list_voice_ids() == ["narrator"]
    assert store.sample_count("narrator") == 2


def test_chat_endpoint_uses_configured_model(monkeypatch):
    monkeypatch.setattr(main, "llm_client", FakeLlmClient())

    response = TestClient(main.app).post("/chat", json={"text": "hello"})

    assert response.status_code == 200
    assert response.json()["model"] == "gemma4:e4b"
    assert response.json()["response"] == "reply to hello"


def test_health_reports_chatterbox_runtime(monkeypatch):
    monkeypatch.setattr(main, "tts", FakeTts())

    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json()["tts_runtime"]["provider"] == "chatterbox_streaming"


def test_voice_roundtrip_endpoint_streams_with_mocked_services(monkeypatch):
    monkeypatch.setattr(main, "llm_client", FakeLlmClient())
    monkeypatch.setattr(main, "transcriber", FakeTranscriber())
    monkeypatch.setattr(main, "tts", FakeTts())

    response = TestClient(main.app).post(
        "/voice/roundtrip",
        files={"audio": ("sample.wav", b"fake wav bytes", "audio/wav")},
        data={"voice_id": "default"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["type"] == "chunk"
    assert events[1]["type"] == "final"
    assert events[1]["transcript"] == "hello robit"
    assert events[1]["model"] == "gemma4:e4b"
    assert events[1]["audio_url"] == "/audio/final.wav"


def test_chat_speak_endpoint_streams_with_mocked_services(monkeypatch):
    monkeypatch.setattr(main, "llm_client", FakeLlmClient())
    monkeypatch.setattr(main, "tts", FakeTts())

    response = TestClient(main.app).post(
        "/chat/speak",
        json={"text": "hello", "voice_id": "default"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["type"] == "response"
    assert events[0]["response"] == "reply to hello"
    assert events[0]["model"] == "gemma4:e4b"
    assert events[1]["type"] == "final"
    assert events[1]["response"] == "reply to hello"
    assert events[1]["model"] == "gemma4:e4b"
    assert events[1]["audio_url"] == "/audio/fake.wav"
    assert events[1]["voice_id"] == "default"
    assert events[1]["active_reference_count"] == 1


def test_voice_synthesize_endpoint_streams_with_mocked_tts(monkeypatch):
    monkeypatch.setattr(main, "tts", FakeTts())

    response = TestClient(main.app).post(
        "/voice/synthesize",
        json={"text": "hello", "voice_id": "default"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["type"] == "chunk"
    assert events[0]["audio_url"] == "/audio/chunk.wav"
    assert events[1]["type"] == "final"
    assert events[1]["audio_url"] == "/audio/final.wav"
    assert events[1]["total_chunks"] == 1
