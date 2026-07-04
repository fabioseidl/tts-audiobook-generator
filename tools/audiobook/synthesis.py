"""Synthesize a single part via the TTS API, validating the audio with retries."""

import io
import time
import wave
from pathlib import Path

import requests


def is_valid_wav(data: bytes) -> bool:
    """Return True if data is a readable, non-empty WAV (not corrupted)."""
    if not data or len(data) < 44:
        return False
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() > 0
    except (wave.Error, EOFError, ValueError):
        return False


def synthesize(text, url, voice, language, timeout):
    """Call the TTS API and return the raw response bytes (or None on error)."""
    try:
        resp = requests.post(
            url,
            json={"text": text, "voice": voice, "language": language},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        print(f"    request failed: {exc}")
        return None
    if resp.status_code != 200:
        print(f"    API returned HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    return resp.content


def process_part(part, args):
    """Synthesize one part, validating the audio, with retries.

    Returns the saved file's name on success, or None on failure.
    """
    # 5-digit zero-pad: keeps every filename the same width so lexicographic
    # sort == numeric order (there are >10000 parts, so 4 digits is not enough).
    audio_path = Path(args.output) / f"part_{part['part_id']:05d}.wav"

    if args.skip_existing and audio_path.exists():
        with audio_path.open("rb") as f:
            if is_valid_wav(f.read()):
                print(f"    already done -> {audio_path.name}")
                return audio_path.name

    for attempt in range(1, args.retries + 1):
        data = synthesize(
            part["text"], args.url, args.voice, args.language, args.timeout
        )
        if data is not None and is_valid_wav(data):
            audio_path.write_bytes(data)
            print(f"    saved {audio_path.name} ({len(data)} bytes)")
            return audio_path.name
        print(f"    attempt {attempt}/{args.retries}: corrupted or missing audio")
        if attempt < args.retries:
            time.sleep(args.retry_delay)
    return None
