"""Synthesize parts via the TTS API, validating the audio with retries."""

import io
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def make_session(workers):
    """Build one connection pool for the whole run.

    Reusing connections matters more than it looks: a fresh connection per part
    also re-pays DNS and TCP setup, and the pool must be at least as large as
    the worker count or the threads serialize on a single connection.
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=workers, pool_maxsize=workers
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def synthesize(session, text, url, voice, language, timeout):
    """Call the TTS API and return the raw response bytes (or None on error)."""
    try:
        resp = session.post(
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


def process_part(part, args, session):
    """Synthesize one part, validating the audio, with retries.

    Returns the saved file's name on success, or None on failure.
    """
    # 5-digit zero-pad: keeps every filename the same width so lexicographic
    # sort == numeric order (there are >10000 parts, so 4 digits is not enough).
    audio_path = Path(args.output) / f"part_{part['part_id']:05d}.wav"
    # Parts may be synthesized concurrently, so every line is tagged with its
    # part_id rather than relying on the order of the output.
    tag = f"[{part['part_id']}]"

    if args.skip_existing and audio_path.exists():
        with audio_path.open("rb") as f:
            if is_valid_wav(f.read()):
                print(f"{tag} already done -> {audio_path.name}")
                return audio_path.name

    for attempt in range(1, args.retries + 1):
        data = synthesize(
            session, part["text"], args.url, args.voice, args.language, args.timeout
        )
        if data is not None and is_valid_wav(data):
            audio_path.write_bytes(data)
            print(f"{tag} saved {audio_path.name} ({len(data)} bytes)")
            return audio_path.name
        print(f"{tag} attempt {attempt}/{args.retries}: corrupted or missing audio")
        if attempt < args.retries:
            time.sleep(args.retry_delay)
    return None


def _record(part, audio_file):
    """Write the outcome of one part back onto it. Returns True on success."""
    if audio_file is None:
        part["status"] = "failed"
        part["audio_file"] = None
        return False
    part["status"] = "done"
    part["audio_file"] = audio_file
    return True


def synthesize_parts(parts, args):
    """Synthesize every part, up to ``args.workers`` at a time.

    The server holds a pool of model instances, so sending several parts at
    once is what keeps the GPU busy; one request at a time leaves it ~25% used.
    Returns the first part that failed, or None if all of them succeeded.
    """
    session = make_session(max(args.workers, 1))

    if args.workers <= 1:
        for part in parts:
            if not _record(part, process_part(part, args, session)):
                return part
        return None

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_part, part, args, session): part for part in parts
        }
        for future in as_completed(futures):
            # Everything queued behind a failure is cancelled below; those
            # futures still come back here, with no result to record.
            if future.cancelled():
                continue
            part = futures[future]
            if not _record(part, future.result()):
                # Don't burn hours on the remaining parts once one is
                # unrecoverable. Parts already in flight are left to finish.
                failures.append(part)
                pool.shutdown(wait=False, cancel_futures=True)
    # Parts finish out of order, so report the earliest one to resume from.
    return min(failures, key=lambda p: p["part_id"]) if failures else None
