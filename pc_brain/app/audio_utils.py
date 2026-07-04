import re
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile


SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac"}


def clean_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip(".-")
    if not cleaned:
        raise HTTPException(status_code=400, detail="voice_id cannot be empty")
    return cleaned


def ensure_data_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


async def save_upload(upload: UploadFile, target_dir: Path, prefix: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type '{suffix or 'unknown'}'. Use MP3, WAV, M4A, WebM, OGG, or FLAC.",
        )

    ensure_data_dirs(target_dir)
    target = target_dir / f"{prefix}-{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            output.write(chunk)
    return target


def normalize_to_wav(source: Path, target: Path) -> Path:
    if not shutil.which("ffmpeg"):
        raise HTTPException(
            status_code=500,
            detail="ffmpeg is required to normalize voice samples. Install ffmpeg and ensure it is on PATH.",
        )

    ensure_data_dirs(target.parent)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "22050",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Could not convert audio to WAV: {completed.stderr.strip()}",
        )
    return target


def split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]
