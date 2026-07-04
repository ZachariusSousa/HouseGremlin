from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from .config import Settings


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    duration_seconds: float | None


class FasterWhisperTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    def _load_model(self):
        if self.settings.stt_provider != "faster_whisper":
            raise HTTPException(status_code=501, detail="Only faster-whisper is supported for v1.")
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="faster-whisper is not installed. Run pip install -r requirements.txt.",
                ) from exc

            self._model = WhisperModel(
                self.settings.stt_model,
                device=self.settings.stt_device,
                compute_type=self.settings.stt_compute_type,
            )
        return self._model

    def warmup(self) -> None:
        try:
            self._load_model()
        except HTTPException:
            return

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        model = self._load_model()
        segments, info = model.transcribe(str(audio_path), vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            raise HTTPException(status_code=400, detail="No speech was detected in the uploaded audio.")
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            duration_seconds=getattr(info, "duration", None),
        )
