from dataclasses import dataclass

from fastapi.testclient import TestClient

from app import main


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


class FakeLlmClient:
    async def chat(self, text: str):
        return FakeChatResult(response=f"reply to {text}", model="gemma4:e4b")


class FakeTranscriber:
    def transcribe(self, path):
        return FakeTranscriptionResult(text="hello robit")


class FakeTts:
    def synthesize(self, text: str, voice_id: str):
        return FakeSynthesisResult(
            audio_url="/audio/fake.wav",
            audio_urls=["/audio/fake.wav"],
            voice_id=voice_id,
        )


def test_chat_endpoint_uses_configured_model(monkeypatch):
    monkeypatch.setattr(main, "llm_client", FakeLlmClient())

    response = TestClient(main.app).post("/chat", json={"text": "hello"})

    assert response.status_code == 200
    assert response.json()["model"] == "gemma4:e4b"
    assert response.json()["response"] == "reply to hello"


def test_voice_roundtrip_endpoint_with_mocked_services(monkeypatch):
    monkeypatch.setattr(main, "llm_client", FakeLlmClient())
    monkeypatch.setattr(main, "transcriber", FakeTranscriber())
    monkeypatch.setattr(main, "tts", FakeTts())

    response = TestClient(main.app).post(
        "/voice/roundtrip",
        files={"audio": ("sample.wav", b"fake wav bytes", "audio/wav")},
        data={"voice_id": "default"},
    )

    assert response.status_code == 200
    assert response.json()["transcript"] == "hello robit"
    assert response.json()["model"] == "gemma4:e4b"
    assert response.json()["audio_url"] == "/audio/fake.wav"


def test_chat_speak_endpoint_with_mocked_services(monkeypatch):
    monkeypatch.setattr(main, "llm_client", FakeLlmClient())
    monkeypatch.setattr(main, "tts", FakeTts())

    response = TestClient(main.app).post(
        "/chat/speak",
        json={"text": "hello", "voice_id": "default"},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "reply to hello"
    assert response.json()["model"] == "gemma4:e4b"
    assert response.json()["audio_url"] == "/audio/fake.wav"
    assert response.json()["voice_id"] == "default"
