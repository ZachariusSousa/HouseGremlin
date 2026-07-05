import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from .audio_utils import clean_spoken_text, ensure_data_dirs, wav_duration_seconds
from .config import Settings
from .timing import timed
from .voices import VoiceStore

SUPPORTED_PROVIDER = "chatterbox_turbo"
MIN_REFERENCE_SECONDS = 5.0


@dataclass
class SynthesisResult:
    audio_url: str
    audio_urls: list[str]
    voice_id: str
    spoken_text: str
    tts_input_chars: int
    active_reference_count: int


class ChatterboxTurboSynthesizer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._voice_refs: dict[str, list[Path]] = {}
        self._voice_store = VoiceStore(settings)

    def _load_model(self):
        if self.settings.tts_provider != SUPPORTED_PROVIDER:
            raise HTTPException(
                status_code=501,
                detail="Only Chatterbox-Turbo is supported for TTS.",
            )
        if self._model is None:
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Chatterbox-Turbo is not installed. Run pip install -r requirements.txt.",
                ) from exc

            try:
                with timed(
                    "tts.chatterbox.load_model",
                    model=self.settings.tts_model,
                    device=self.settings.tts_device,
                ):
                    self._configure_torch_runtime()
                    self._model = ChatterboxTurboTTS.from_pretrained(device=self.settings.tts_device)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Chatterbox-Turbo failed to load. Recreate the Python 3.11 venv "
                        "and run pip install -r requirements.txt. "
                        f"Original error: {type(exc).__name__}: {exc}"
                    ),
                ) from exc
        return self._model

    @staticmethod
    def _configure_torch_runtime() -> None:
        try:
            import torch
        except ImportError:
            return

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

    @staticmethod
    def _inference_context():
        try:
            import torch
        except ImportError:
            return nullcontext()
        return torch.inference_mode()

    @staticmethod
    def _save_audio(path: Path, wav, sample_rate: int) -> None:
        try:
            import torchaudio
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="torchaudio is not installed. Run pip install -r requirements.txt.",
            ) from exc

        torchaudio.save(str(path), wav, sample_rate)

    def runtime_info(self) -> dict:
        info = {
            "provider": SUPPORTED_PROVIDER,
            "model": self.settings.tts_model,
            "configured_device": self.settings.tts_device,
            "model_loaded": self._model is not None,
            "temperature": self.settings.tts_temperature,
            "top_p": self.settings.tts_top_p,
            "top_k": self.settings.tts_top_k,
            "repetition_penalty": self.settings.tts_repetition_penalty,
            "norm_loudness": self.settings.tts_norm_loudness,
        }
        if self._model is not None:
            info["sample_rate"] = getattr(self._model, "sr", None)

        try:
            import torch
        except ImportError:
            info["torch_available"] = False
            return info

        info.update(
            {
                "torch_available": True,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "cuda_device_count": torch.cuda.device_count(),
            }
        )
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        return info

    def warmup(self, synthesize_default_voice: bool = True) -> None:
        try:
            self._load_model()
            if synthesize_default_voice and self._voice_store.reference_paths(self.settings.voice_id):
                self.synthesize("Ready.", self.settings.voice_id)
        except Exception:
            return

    def register_voice(self, voice_id: str, reference_wavs: Path | list[Path]) -> None:
        references = [reference_wavs] if isinstance(reference_wavs, Path) else reference_wavs
        if not references or any(not reference.exists() for reference in references):
            raise HTTPException(status_code=404, detail=f"Voice reference not found for '{voice_id}'.")
        self._voice_refs[voice_id] = references
        self.warmup(synthesize_default_voice=False)

    def _references_for(self, voice_id: str) -> list[Path]:
        if voice_id in self._voice_refs:
            return self._voice_refs[voice_id]
        references = self._voice_store.reference_paths(voice_id)
        if not references:
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{voice_id}' has no reference sample. Upload one with POST /voices.",
            )
        self._voice_refs[voice_id] = references
        return references

    def _active_reference_for(self, voice_id: str) -> Path:
        references = self._references_for(voice_id)
        ordered = sorted(references, key=lambda reference: (reference.name != "reference.wav", reference.name))
        for reference in ordered:
            try:
                duration_seconds = wav_duration_seconds(reference)
            except Exception:
                continue
            if duration_seconds > MIN_REFERENCE_SECONDS:
                return reference
        raise HTTPException(
            status_code=400,
            detail=(
                f"Voice '{voice_id}' needs a reference sample longer than {MIN_REFERENCE_SECONDS:.0f} seconds "
                "for Chatterbox-Turbo. Upload a longer clip with POST /voices."
            ),
        )

    def synthesize(self, text: str, voice_id: str) -> SynthesisResult:
        with timed("tts.clean_text", voice_id=voice_id, text_chars=len(text)):
            stripped = clean_spoken_text(text)
        if not stripped:
            raise HTTPException(status_code=400, detail="text does not contain anything speakable")

        with timed("tts.synthesize.total", voice_id=voice_id, text_chars=len(stripped)):
            model = self._load_model()
            with timed("tts.reference_select", voice_id=voice_id):
                reference = self._active_reference_for(voice_id)
            ensure_data_dirs(self.settings.audio_dir)

            final_path = self.settings.audio_dir / f"{uuid.uuid4().hex}.wav"
            with timed(
                "tts.chatterbox.generate",
                voice_id=voice_id,
                text_chars=len(stripped),
                active_reference_count=1,
                temperature=self.settings.tts_temperature,
                top_p=self.settings.tts_top_p,
                top_k=self.settings.tts_top_k,
                repetition_penalty=self.settings.tts_repetition_penalty,
            ):
                with self._inference_context():
                    try:
                        wav = model.generate(
                            stripped,
                            audio_prompt_path=str(reference),
                            temperature=self.settings.tts_temperature,
                            top_p=self.settings.tts_top_p,
                            top_k=self.settings.tts_top_k,
                            repetition_penalty=self.settings.tts_repetition_penalty,
                            norm_loudness=self.settings.tts_norm_loudness,
                        )
                    except AssertionError as exc:
                        raise HTTPException(status_code=400, detail=str(exc)) from exc
                self._save_audio(final_path, wav, model.sr)

        return SynthesisResult(
            audio_url=f"/audio/{final_path.name}",
            audio_urls=[f"/audio/{final_path.name}"],
            voice_id=voice_id,
            spoken_text=stripped,
            tts_input_chars=len(stripped),
            active_reference_count=1,
        )
