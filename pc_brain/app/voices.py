import uuid
from pathlib import Path

from .audio_utils import clean_id, normalize_to_wav
from .config import Settings


class VoiceStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    def reference_path(self, voice_id: str) -> Path:
        return self.settings.voices_dir / clean_id(voice_id) / "reference.wav"

    def reference_paths(self, voice_id: str) -> list[Path]:
        voice_dir = self.settings.voices_dir / clean_id(voice_id)
        if not voice_dir.exists():
            return []
        return sorted(voice_dir.glob("reference*.wav"))

    def save_reference(self, voice_id: str, source_audio: Path) -> tuple[str, Path, list[Path]]:
        cleaned = clean_id(voice_id)
        voice_dir = self.settings.voices_dir / cleaned
        reference = self.reference_path(cleaned)
        if reference.exists():
            reference = voice_dir / f"reference-{uuid.uuid4().hex}.wav"
        normalize_to_wav(source_audio, reference)
        return cleaned, reference, self.reference_paths(cleaned)

    def list_voice_ids(self) -> list[str]:
        if not self.settings.voices_dir.exists():
            return []
        return sorted(
            path.name
            for path in self.settings.voices_dir.iterdir()
            if path.is_dir() and self.reference_paths(path.name)
        )

    def sample_count(self, voice_id: str) -> int:
        return len(self.reference_paths(voice_id))
