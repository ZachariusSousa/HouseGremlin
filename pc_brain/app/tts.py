import inspect
import uuid
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from fastapi import HTTPException

from .audio_utils import clean_spoken_text, ensure_data_dirs, wav_duration_seconds
from .config import Settings
from .cuda_paths import configure_windows_cuda_dll_paths
from .timing import timed
from .voices import VoiceStore

PROVIDER = "chatterbox_streaming"
MIN_REFERENCE_SECONDS = 5.0


@dataclass
class SynthesisResult:
    audio_url: str
    audio_urls: list[str]
    voice_id: str
    spoken_text: str
    tts_input_chars: int
    active_reference_count: int


@dataclass
class SynthesisStreamEvent:
    event: dict
    final_result: SynthesisResult | None = None


class ChatterboxSynthesizer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._voice_refs: dict[str, list[Path]] = {}
        self._voice_store = VoiceStore(settings)

    def _load_model(self):
        if self._model is None:
            try:
                configure_windows_cuda_dll_paths()
                from chatterbox.tts import ChatterboxTTS
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Chatterbox Streaming could not be imported. "
                        "Run pip install -r requirements.txt. "
                        f"Original error: {type(exc).__name__}: {exc}"
                    ),
                ) from exc

            try:
                with timed(
                    "tts.chatterbox.load_model",
                    model=self.settings.tts_model,
                    device=self.settings.tts_device,
                ):
                    self._configure_torch_runtime()
                    self._model = ChatterboxTTS.from_pretrained(device=self.settings.tts_device)
                    if not hasattr(self._model, "generate_stream"):
                        installed = self._installed_chatterbox_package()
                        raise RuntimeError(
                            "Installed Chatterbox package does not provide generate_stream. "
                            f"Found {installed}. Run: pip uninstall -y chatterbox-tts chatterbox-streaming; "
                            "pip install --no-deps chatterbox-streaming==0.1.2"
                        )
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Chatterbox Streaming failed to load. Recreate the Python 3.11 venv "
                        "and run pip install -r requirements.txt. "
                        f"Original error: {type(exc).__name__}: {exc}"
                    ),
                ) from exc
        return self._model

    @staticmethod
    def _installed_chatterbox_package() -> str:
        installed = []
        for package in ("chatterbox-streaming", "chatterbox-tts"):
            try:
                installed.append(f"{package}=={metadata.version(package)}")
            except metadata.PackageNotFoundError:
                continue
        return ", ".join(installed) if installed else "no chatterbox distribution metadata"

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
            "provider": PROVIDER,
            "model": self.settings.tts_model,
            "configured_device": self.settings.tts_device,
            "model_loaded": self._model is not None,
            "temperature": self.settings.tts_temperature,
            "top_p": self.settings.tts_top_p,
            "repetition_penalty": self.settings.tts_repetition_penalty,
            "norm_loudness": self.settings.tts_norm_loudness,
            "chunk_size": self.settings.tts_chunk_size,
            "exaggeration": self.settings.tts_exaggeration,
            "cfg_weight": self.settings.tts_cfg_weight,
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
                "for Chatterbox Streaming. Upload a longer clip with POST /voices."
            ),
        )

    def _clean_text_or_raise(self, text: str, voice_id: str) -> str:
        with timed("tts.clean_text", voice_id=voice_id, text_chars=len(text)):
            stripped = clean_spoken_text(text)
        if not stripped:
            raise HTTPException(status_code=400, detail="text does not contain anything speakable")
        return stripped

    def _generate_kwargs(self, reference: Path) -> dict:
        return {
            "audio_prompt_path": str(reference),
            "temperature": self.settings.tts_temperature,
            "top_p": self.settings.tts_top_p,
            "repetition_penalty": self.settings.tts_repetition_penalty,
            "norm_loudness": self.settings.tts_norm_loudness,
            "exaggeration": self.settings.tts_exaggeration,
            "cfg_weight": self.settings.tts_cfg_weight,
        }

    @staticmethod
    def _supported_kwargs(callable_obj, kwargs: dict) -> dict:
        signature = inspect.signature(callable_obj)
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in signature.parameters}

    def synthesize(self, text: str, voice_id: str) -> SynthesisResult:
        stripped = self._clean_text_or_raise(text, voice_id)

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
                repetition_penalty=self.settings.tts_repetition_penalty,
                exaggeration=self.settings.tts_exaggeration,
                cfg_weight=self.settings.tts_cfg_weight,
            ):
                with self._inference_context():
                    try:
                        wav = model.generate(
                            stripped,
                            **self._supported_kwargs(model.generate, self._generate_kwargs(reference)),
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

    def synthesize_stream(self, text: str, voice_id: str) -> Iterator[SynthesisStreamEvent]:
        stripped = self._clean_text_or_raise(text, voice_id)
        model = self._load_model()
        with timed("tts.reference_select", voice_id=voice_id):
            reference = self._active_reference_for(voice_id)
        ensure_data_dirs(self.settings.audio_dir)

        def events() -> Iterator[SynthesisStreamEvent]:
            chunk_urls: list[str] = []
            chunks = []
            first_latency_seconds = None
            final_path = self.settings.audio_dir / f"{uuid.uuid4().hex}.wav"

            with timed(
                "tts.chatterbox.generate_stream",
                voice_id=voice_id,
                text_chars=len(stripped),
                active_reference_count=1,
                chunk_size=self.settings.tts_chunk_size,
                temperature=self.settings.tts_temperature,
                top_p=self.settings.tts_top_p,
                repetition_penalty=self.settings.tts_repetition_penalty,
                exaggeration=self.settings.tts_exaggeration,
                cfg_weight=self.settings.tts_cfg_weight,
            ):
                with self._inference_context():
                    try:
                        stream = model.generate_stream(
                            stripped,
                            **self._supported_kwargs(
                                model.generate_stream,
                                {
                                    **self._generate_kwargs(reference),
                                    "chunk_size": self.settings.tts_chunk_size,
                                },
                            ),
                        )
                        for chunk_index, (chunk, metrics) in enumerate(stream, start=1):
                            chunks.append(chunk)
                            chunk_path = self.settings.audio_dir / f"{uuid.uuid4().hex}.wav"
                            self._save_audio(chunk_path, chunk, model.sr)
                            chunk_url = f"/audio/{chunk_path.name}"
                            chunk_urls.append(chunk_url)
                            first_latency_seconds = getattr(metrics, "latency_to_first_chunk", first_latency_seconds)
                            yield SynthesisStreamEvent(
                                event={
                                    "type": "chunk",
                                    "chunk_index": chunk_index,
                                    "audio_url": chunk_url,
                                    "voice_id": voice_id,
                                    "first_latency_seconds": first_latency_seconds,
                                }
                            )
                    except AssertionError as exc:
                        raise HTTPException(status_code=400, detail=str(exc)) from exc

            if not chunks:
                raise HTTPException(status_code=500, detail="Chatterbox Streaming did not yield any audio chunks.")

            try:
                import torch
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="torch is not installed. Run pip install -r requirements.txt.",
                ) from exc

            final_wav = torch.cat(chunks, dim=-1)
            self._save_audio(final_path, final_wav, model.sr)
            final_result = SynthesisResult(
                audio_url=f"/audio/{final_path.name}",
                audio_urls=chunk_urls,
                voice_id=voice_id,
                spoken_text=stripped,
                tts_input_chars=len(stripped),
                active_reference_count=1,
            )
            yield SynthesisStreamEvent(
                event={
                    "type": "final",
                    "audio_url": final_result.audio_url,
                    "audio_urls": final_result.audio_urls,
                    "voice_id": final_result.voice_id,
                    "spoken_text": final_result.spoken_text,
                    "tts_input_chars": final_result.tts_input_chars,
                    "active_reference_count": final_result.active_reference_count,
                    "total_chunks": len(chunk_urls),
                },
                final_result=final_result,
            )

        return events()
