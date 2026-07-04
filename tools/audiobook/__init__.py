"""Turn an ebook markdown file into an audiobook via the XTTS server.

The pipeline is split by concern:
  - text:      clean markdown and break it into small parts
  - synthesis: call the TTS API for each part, validate and retry
  - join:      concatenate the per-part WAVs into grouped MP3 files

See ``tools/generate_audiobook.py`` for the CLI entry point.
"""
