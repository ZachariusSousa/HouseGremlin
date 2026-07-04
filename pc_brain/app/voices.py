from pathlib import Path

from .audio_utils import clean_id, normalize_to_wav
from .config import Settings


class VoiceStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    def reference_path(self, voice_id: str) -> Path:
        return self.settings.voices_dir / clean_id(voice_id) / "reference.wav"

    def save_reference(self, voice_id: str, source_audio: Path) -> tuple[str, Path]:
        cleaned = clean_id(voice_id)
        reference = self.reference_path(cleaned)
        normalize_to_wav(source_audio, reference)
        return cleaned, reference
