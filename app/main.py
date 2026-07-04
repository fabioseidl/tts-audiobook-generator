"""Split an ebook markdown file into small parts and synthesize each with the
XTTS server, saving one audio file per part.

Pipeline:
  1. Read the markdown (default: ebook/ebook_full.md).
  2. Clean markdown syntax and normalize whitespace.
  3. Break it into an ordered array of parts, each <= MAX_CHARS characters,
     split on sentence / clause boundaries so the audio sounds natural.
  4. POST each part's text to the TTS API and save the returned WAV.
  5. Verify the WAV is not corrupted before moving on to the next part; retry
     a few times, and stop with a clear error if a part cannot be produced.

The parts array (part_id + text + audio_file + status) is also written to
output/parts.json so the run can be inspected or resumed.

Usage:
    python app/main.py                     # full run
    python app/main.py --dry-run           # only build parts.json
    python app/main.py --limit 5           # only the first 5 parts
    python app/main.py --start-id 42       # resume from part 42
    python app/main.py --join              # join WAV parts into grouped MP3s

The --join mode is a separate step run after synthesis: it concatenates the
per-part WAVs in output/audio into MP3 files under output/audiobook, grouped
JOIN_GROUP_SIZE parts to a file (parts 1-1000 -> audiobook_part_1.mp3,
1001-2000 -> audiobook_part_2.mp3, ...).
"""

import argparse
import io
import json
import re
import sys
import time
import wave
from pathlib import Path

import requests

# --- defaults -------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "ebook" / "ebook_full.md"
DEFAULT_OUTPUT = ROOT / "output/audio"
DEFAULT_AUDIOBOOK = ROOT / "output/audiobook"
DEFAULT_URL = "http://localhost:8000/tts"
DEFAULT_VOICE = "narrador.wav"
DEFAULT_LANGUAGE = "pt"
MAX_CHARS = 200
# --join groups this many parts (by part_id) into each MP3, and encodes at
# this CBR bitrate (kbps).
JOIN_GROUP_SIZE = 1000
MP3_BITRATE = 128

# Sentence terminators (incl. closing quotes/paren that may trail them).
# Alternation of fixed-width lookbehinds: Python forbids a variable-width one.
_SENTENCE_END = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][\"'”’\)\]]))\s+")
# Clause boundaries used to break a sentence that is longer than MAX_CHARS.
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:—–])\s+")


# --- markdown cleaning ----------------------------------------------------

def clean_markdown(text: str) -> str:
    """Strip common markdown syntax so only spoken text remains."""
    # Code fences and inline code.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Images and links -> keep the visible text.
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Heading markers, blockquotes, list bullets at line start.
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)
    text = re.sub(r"(?m)^\s{0,3}[-*+]\s+", "", text)
    # Horizontal rules.
    text = re.sub(r"(?m)^\s*([-*_])\1{2,}\s*$", "", text)
    # Emphasis markers (leave the words, drop the surrounding * or _).
    text = re.sub(r"(\*\*|__|\*|_)(.+?)\1", r"\2", text)
    return text


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Split a too-long sentence into <= max_chars pieces.

    Tries clause boundaries first, then falls back to splitting on spaces so no
    piece ever exceeds the limit.
    """
    pieces: list[str] = []
    for clause in _CLAUSE_SPLIT.split(sentence):
        clause = clause.strip()
        if not clause:
            continue
        if len(clause) <= max_chars:
            pieces.append(clause)
            continue
        # Still too long: wrap on word boundaries.
        current = ""
        for word in clause.split():
            if len(word) > max_chars:
                # A single monstrous token; chop it by characters.
                if current:
                    pieces.append(current)
                    current = ""
                for i in range(0, len(word), max_chars):
                    pieces.append(word[i:i + max_chars])
                continue
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= max_chars:
                current += " " + word
            else:
                pieces.append(current)
                current = word
        if current:
            pieces.append(current)
    return pieces


def build_parts(markdown: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """Turn cleaned markdown into an ordered list of parts.

    Each returned dict has: part_id, text, chars.
    """
    # Split into blank-line-separated blocks first.
    raw_blocks = re.split(r"\n\s*\n", markdown)

    blocks: list[str] = []
    for block in raw_blocks:
        block = clean_markdown(block)
        # Join wrapped lines and collapse whitespace within the block.
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        # PDF-extracted text sometimes breaks a single sentence across blank
        # lines; if the previous block did not end a sentence, glue this one on.
        if blocks and not re.search(r"[.!?…][\"'”’\)\]]?$", blocks[-1]):
            blocks[-1] = (blocks[-1] + " " + block).strip()
        else:
            blocks.append(block)

    # Break each block into sentences, then greedily pack them into chunks.
    chunks: list[str] = []
    for block in blocks:
        current = ""
        for sentence in _SENTENCE_END.split(block):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(_hard_split(sentence, max_chars))
                continue
            if not current:
                current = sentence
            elif len(current) + 1 + len(sentence) <= max_chars:
                current += " " + sentence
            else:
                chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)

    parts = []
    for i, text in enumerate(chunks, start=1):
        # The TTS text must contain no periods; replace every "." with ",".
        # Done after splitting so sentence boundaries are still detected above.
        text = text.replace(".", ",")
        parts.append({"part_id": i, "text": text, "chars": len(text)})
    return parts


# --- TTS + validation -----------------------------------------------------

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


# --- joining parts into MP3 ------------------------------------------------

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


# --- main -----------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Markdown file to read.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory to save audio parts and parts.json.")
    parser.add_argument("--url", default=DEFAULT_URL, help="TTS API endpoint.")
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS,
                        help="Maximum characters per part.")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3,
                        help="Attempts per part before giving up.")
    parser.add_argument("--retry-delay", type=float, default=2.0,
                        help="Seconds to wait between retries.")
    parser.add_argument("--start-id", type=int, default=1,
                        help="Skip parts with a lower part_id (resume).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process this many parts.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip parts whose audio file already exists and is valid.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only build parts.json; do not call the TTS API.")
    parser.add_argument("--join", action="store_true",
                        help="Join the WAV parts in --output into grouped MP3 "
                             "files; do not call the TTS API.")
    parser.add_argument("--audiobook-dir", type=Path, default=DEFAULT_AUDIOBOOK,
                        help="Directory for the joined MP3 files (--join).")
    parser.add_argument("--group-size", type=int, default=JOIN_GROUP_SIZE,
                        help="Parts per joined MP3 file (--join).")
    parser.add_argument("--bitrate", type=int, default=MP3_BITRATE,
                        help="MP3 bitrate in kbps (--join).")
    args = parser.parse_args(argv)

    # Joining is a standalone post-processing step; it reads the already
    # synthesized WAVs and needs neither the markdown input nor the TTS server.
    if args.join:
        written = join_audiobook(
            args.output, args.audiobook_dir, args.group_size, args.bitrate
        )
        return 0 if written else 1

    if not args.input.exists():
        parser.error(f"input file not found: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    markdown = args.input.read_text(encoding="utf-8")
    parts = build_parts(markdown, args.max_chars)
    print(f"Built {len(parts)} parts from {args.input} "
          f"(<= {args.max_chars} chars each).")

    parts_json = args.output / "parts.json"

    if args.dry_run:
        parts_json.write_text(
            json.dumps(parts, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Dry run: wrote {parts_json}")
        return 0

    processed = 0
    for part in parts:
        if part["part_id"] < args.start_id:
            part["status"] = "skipped"
            continue
        if args.limit is not None and processed >= args.limit:
            part["status"] = "pending"
            continue

        print(f"[{part['part_id']}/{len(parts)}] {part['chars']} chars: "
              f"{part['text'][:60]!r}")
        audio_file = process_part(part, args)
        processed += 1

        if audio_file is None:
            part["status"] = "failed"
            part["audio_file"] = None
            # Persist progress before bailing out.
            parts_json.write_text(
                json.dumps(parts, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"\nStopping: part {part['part_id']} could not be synthesized "
                  f"after {args.retries} attempts.")
            return 1

        part["status"] = "done"
        part["audio_file"] = audio_file

    parts_json.write_text(
        json.dumps(parts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Wrote {parts_json} and audio to {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
