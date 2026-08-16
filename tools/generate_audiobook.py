"""Split an ebook markdown file into small parts and synthesize each with the
XTTS server, saving one audio file per part.

Pipeline:
  1. Read the markdown (default: ebook/ebook_full.md).
  2. Clean markdown syntax and normalize whitespace.
  3. Break it into an ordered array of parts, each <= MAX_CHARS characters,
     split on sentence / clause boundaries so the audio sounds natural.
  4. POST each part's text to the TTS API and save the returned WAV, several
     parts at a time (see --workers).
  5. Verify the WAV is not corrupted before moving on to the next part; retry
     a few times, and stop with a clear error if a part cannot be produced.

The parts array (part_id + text + audio_file + status) is also written to
output/parts.json so the run can be inspected or resumed.

Usage:
    python tools/generate_audiobook.py                # full run
    python tools/generate_audiobook.py --dry-run      # only build parts.json
    python tools/generate_audiobook.py --limit 5      # only the first 5 parts
    python tools/generate_audiobook.py --start-id 42  # resume from part 42
    python tools/generate_audiobook.py --workers 4    # 4 concurrent requests
    python tools/generate_audiobook.py --join         # join WAV parts into MP3s

The --join mode is a separate step run after synthesis: it concatenates the
per-part WAVs in output/audio into MP3 files under output/audiobook, grouped
JOIN_GROUP_SIZE parts to a file (parts 1-1000 -> audiobook_part_1.mp3,
1001-2000 -> audiobook_part_2.mp3, ...).

The implementation is split by concern under the ``audiobook`` package:
text cleaning/chunking, synthesis, and joining.
"""

import argparse
import json
import sys
from pathlib import Path

from audiobook.text import build_parts, MAX_CHARS
from audiobook.synthesis import synthesize_parts
from audiobook.join import join_audiobook, JOIN_GROUP_SIZE, MP3_BITRATE

# --- defaults -------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "ebook" / "ebook_full.md"
DEFAULT_OUTPUT = ROOT / "output/audio"
DEFAULT_AUDIOBOOK = ROOT / "output/audiobook"
DEFAULT_URL = "http://localhost:8000/tts"
DEFAULT_VOICE = "narrador.wav"
DEFAULT_LANGUAGE = "pt"
# Matches the server's default TTS_POOL_SIZE: XTTS is autoregressive, so one
# request at a time leaves most of the GPU idle.
DEFAULT_WORKERS = 3


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
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Parts to synthesize concurrently. Keep this at or "
                             "below the server's TTS_POOL_SIZE; higher just "
                             "queues requests. Use 1 for strictly serial runs.")
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

    pending = []
    for part in parts:
        if part["part_id"] < args.start_id:
            part["status"] = "skipped"
            continue
        if args.limit is not None and len(pending) >= args.limit:
            part["status"] = "pending"
            continue
        pending.append(part)

    print(f"Synthesizing {len(pending)} parts, {args.workers} at a time.")
    failed = synthesize_parts(pending, args)

    # Persist progress either way: on failure this is the resume point.
    parts_json.write_text(
        json.dumps(parts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if failed is not None:
        print(f"\nStopping: part {failed['part_id']} could not be synthesized "
              f"after {args.retries} attempts.")
        print(f"Resume with --start-id {failed['part_id']} --skip-existing")
        return 1

    print(f"\nDone. Wrote {parts_json} and audio to {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
