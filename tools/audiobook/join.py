"""Concatenate the per-part WAVs into grouped MP3 files.

This is a standalone post-processing step: it reads the already synthesized
WAVs and needs neither the markdown input nor the TTS server.
"""

import re
import wave
from pathlib import Path

# --join groups this many parts (by part_id) into each MP3, and encodes at
# this CBR bitrate (kbps).
JOIN_GROUP_SIZE = 1000
MP3_BITRATE = 128


def _part_id_from_name(path: Path):
    """Extract the numeric part id from a filename like part_00042.wav."""
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None


def join_audiobook(audio_dir: Path, out_dir: Path,
                   group_size: int = JOIN_GROUP_SIZE,
                   bitrate: int = MP3_BITRATE) -> int:
    """Concatenate the per-part WAVs into grouped MP3 files.

    Parts are bucketed by part_id, so parts 1..group_size become
    audiobook_part_1.mp3, the next group_size become audiobook_part_2.mp3, and
    so on -- gaps (missing parts) do not shift a part into another group.
    Returns the number of MP3 files written.
    """
    try:
        import lameenc
    except ImportError:
        print("The --join feature needs the 'lameenc' package: "
              "pip install lameenc")
        return 0

    wavs = []
    for path in audio_dir.glob("part_*.wav"):
        pid = _part_id_from_name(path)
        if pid is not None:
            wavs.append((pid, path))
    wavs.sort()

    if not wavs:
        print(f"No WAV parts found in {audio_dir}")
        return 0

    # Bucket parts into groups of `group_size` by part_id (1-indexed).
    groups: dict[int, list[tuple[int, Path]]] = {}
    for pid, path in wavs:
        group_no = (pid - 1) // group_size + 1
        groups.setdefault(group_no, []).append((pid, path))

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Joining {len(wavs)} parts into {len(groups)} MP3 file(s) "
          f"({group_size} parts per file) in {out_dir}/")

    written = 0
    for group_no in sorted(groups):
        members = groups[group_no]
        params = None  # (channels, sampwidth, framerate) of the first part
        pcm = bytearray()
        for pid, path in members:
            with wave.open(str(path)) as wav:
                fmt = (wav.getnchannels(), wav.getsampwidth(),
                       wav.getframerate())
                if params is None:
                    params = fmt
                elif fmt != params:
                    print(f"    skipping {path.name}: audio format {fmt} "
                          f"differs from {params}")
                    continue
                pcm += wav.readframes(wav.getnframes())

        channels, sampwidth, framerate = params
        if sampwidth != 2:
            print(f"  group {group_no}: {sampwidth * 8}-bit audio is not "
                  f"supported (need 16-bit PCM); skipping.")
            continue

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(bitrate)
        encoder.set_in_sample_rate(framerate)
        encoder.set_channels(channels)
        encoder.set_quality(2)  # 2 = high quality, slower; 7 = fast
        mp3 = encoder.encode(bytes(pcm))
        mp3 += encoder.flush()

        out_path = out_dir / f"audiobook_part_{group_no}.mp3"
        out_path.write_bytes(mp3)
        first_id = members[0][0]
        last_id = members[-1][0]
        print(f"  wrote {out_path.name}: parts {first_id}-{last_id} "
              f"({len(members)} files, {len(mp3):,} bytes)")
        written += 1

    print(f"\nDone. Wrote {written} MP3 file(s) to {out_dir}/")
    return written
