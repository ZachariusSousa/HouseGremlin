import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from .audio_utils import ensure_data_dirs, split_sentences
from .config import Settings


@dataclass
class SynthesisResult:
    audio_url: str
    audio_urls: list[str]
    voice_id: str


class XttsSynthesizer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._voice_refs: dict[str, Path] = {}

    def _load_model(self):
        if self.settings.tts_provider != "xtts":
            raise HTTPException(status_code=501, detail="Only XTTS is supported for v1.")
        if self._model is None:
            try:
                from TTS.api import TTS
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Coqui TTS is not installed. Run pip install -r requirements.txt.",
                ) from exc

            try:
                self._model = TTS(self.settings.tts_model)
                if self.settings.tts_device == "cuda":
                    self._model = self._model.to("cuda")
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "XTTS failed to load. Reinstall dependencies with "
                        "pip install -r requirements.txt, and ensure transformers==4.33.3. "
                        f"Original error: {type(exc).__name__}: {exc}"
                    ),
                ) from exc
        return self._model

    def warmup(self) -> None:
        try:
            self._load_model()
        except Exception:
            return

    def register_voice(self, voice_id: str, reference_wav: Path) -> None:
        if not reference_wav.exists():
            raise HTTPException(status_code=404, detail=f"Voice reference not found for '{voice_id}'.")
        self._voice_refs[voice_id] = reference_wav
        self.warmup()

    def _reference_for(self, voice_id: str) -> Path:
        if voice_id in self._voice_refs:
            return self._voice_refs[voice_id]
        reference = self.settings.voices_dir / voice_id / "reference.wav"
        if not reference.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{voice_id}' has no reference sample. Upload one with POST /voices.",
            )
        self._voice_refs[voice_id] = reference
        return reference

    def synthesize(self, text: str, voice_id: str) -> SynthesisResult:
        stripped = text.strip()
        if not stripped:
            raise HTTPException(status_code=400, detail="text cannot be empty")

        model = self._load_model()
        reference = self._reference_for(voice_id)
        ensure_data_dirs(self.settings.audio_dir)

        sentences = split_sentences(stripped) or [stripped]
        chunk_paths = []
        for index, sentence in enumerate(sentences):
            chunk_path = self.settings.audio_dir / f"{uuid.uuid4().hex}-{index}.wav"
            model.tts_to_file(
                text=sentence,
                speaker_wav=str(reference),
                language=self.settings.tts_language,
                file_path=str(chunk_path),
            )
            chunk_paths.append(chunk_path)

        if len(chunk_paths) == 1:
            final_path = chunk_paths[0]
        else:
            final_path = self.settings.audio_dir / f"{uuid.uuid4().hex}.wav"
            self._concat_wavs(chunk_paths, final_path)

        return SynthesisResult(
            audio_url=f"/audio/{final_path.name}",
            audio_urls=[f"/audio/{path.name}" for path in chunk_paths],
            voice_id=voice_id,
        )

    @staticmethod
    def _concat_wavs(paths: list[Path], target: Path) -> None:
        with wave.open(str(paths[0]), "rb") as first:
            params = first.getparams()
            frames = [first.readframes(first.getnframes())]

        for path in paths[1:]:
            with wave.open(str(path), "rb") as current:
                if current.getparams()[:3] != params[:3]:
                    raise HTTPException(status_code=500, detail="Could not combine generated TTS chunks.")
                frames.append(current.readframes(current.getnframes()))

        with wave.open(str(target), "wb") as output:
            output.setparams(params)
            for frame_data in frames:
                output.writeframes(frame_data)
