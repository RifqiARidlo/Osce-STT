from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import HTTPException, Request, UploadFile

from app.core.services import Services

CHUNK = 1 << 20
ALLOWED_EXT = {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".flac", ".mp4", ".opus"}


def get_services(request: Request) -> Services:
    return request.app.state.services


async def save_upload(upload: UploadFile, dest_stem: Path, max_mb: int) -> Path:
    """Simpan UploadFile ke disk secara streaming dengan batas ukuran."""
    ext = Path(upload.filename or "audio.wav").suffix.lower() or ".wav"
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"Ekstensi {ext} tidak didukung. Gunakan: {sorted(ALLOWED_EXT)}")
    dest = dest_stem.with_suffix(ext)
    limit = max_mb * CHUNK
    written = 0
    try:
        with dest.open("wb") as fh:
            while chunk := await upload.read(CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"File melebihi {max_mb} MB.")
                fh.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "File audio kosong.")
    return dest


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None
