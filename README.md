# XTTS Server

A small FastAPI service that wraps [XTTS v2](https://huggingface.co/coqui/XTTS-v2)
for text-to-speech with voice cloning. Runs on GPU inside Docker.

Built on the maintained [`coqui-tts`](https://pypi.org/project/coqui-tts/)
package (the idiap fork of the original, now-abandoned Coqui `TTS`), which
supports Python 3.12 and current PyTorch.

## Requirements

- NVIDIA GPU with a recent driver
- Docker + the NVIDIA Container Toolkit (so `--gpus` / `deploy.devices` works)

## Layout

```
voices/   reference speaker .wav clips (bind-mounted, read at request time)
output/   generated audio (bind-mounted, files auto-deleted after download)
models/   XTTS model cache (bind-mounted, persisted across restarts)
```

Create the folders and drop at least one reference clip in `voices/` before
starting (a clean 6–20 s mono WAV of the target speaker works well):

```bash
mkdir -p voices output models
# cp my_speaker.wav voices/
```

## Run

```bash
docker compose up --build
```

First start downloads the XTTS model (~1.8 GB) into `models/`; subsequent
starts reuse it. The service listens on port `8000`.

## API

| Method | Path      | Description                          |
|--------|-----------|--------------------------------------|
| GET    | `/health` | Liveness check                       |
| GET    | `/voices` | List available reference voices      |
| POST   | `/tts`    | Synthesize speech, returns a WAV file |

### Example

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Olá, isto é um teste.", "voice": "my_speaker.wav", "language": "pt"}' \
  --output result.wav
```

`language` is a supported XTTS code (`en`, `pt`, `es`, `fr`, `de`, `it`, ...);
defaults to `pt`.

## Audiobook generator (`tools/`)

[`tools/generate_audiobook.py`](tools/generate_audiobook.py) is a client that
turns a whole ebook into an audiobook by driving the TTS server. It reads a
markdown file, strips the markdown syntax, splits the text into small parts on
sentence/clause boundaries (`<= --max-chars`, default 200), and POSTs each part
to the API, saving one WAV per part.

The logic is split by concern under the `tools/audiobook/` package:

```
tools/generate_audiobook.py   CLI + orchestration
tools/audiobook/text.py       clean markdown, chunk into parts
tools/audiobook/synthesis.py  call the TTS API, validate + retry
tools/audiobook/join.py       concatenate WAVs into grouped MP3s
```

Inputs and outputs:

```
ebook/ebook_full.md      input markdown (default)
output/audio/            one part_NNNNN.wav per part
output/audio/parts.json  ordered parts (part_id, text, audio_file, status)
output/audiobook/        grouped MP3s produced by --join
```

Each part's audio is validated as a readable WAV and retried a few times; the
run stops with a clear error if a part can't be produced. Progress is written
to `parts.json`, so a failed or interrupted run can be resumed.

Start the TTS server first (see [Run](#run)), then:

```bash
python tools/generate_audiobook.py                 # full run
python tools/generate_audiobook.py --dry-run       # only build parts.json, no synthesis
python tools/generate_audiobook.py --limit 5       # only the first 5 parts
python tools/generate_audiobook.py --start-id 42   # resume from part 42
python tools/generate_audiobook.py --skip-existing # skip parts already synthesized
python tools/generate_audiobook.py --join          # concatenate WAVs into grouped MP3s
```

Other options: `--input`, `--output`, `--url`, `--voice`, `--language`,
`--timeout`, `--retries`, `--retry-delay`. `--join` is a separate
post-processing step that concatenates the per-part WAVs into MP3 files under
`output/audiobook/`, `--group-size` parts each (default 1000, so parts 1–1000
become `audiobook_part_1.mp3`, and so on). It needs the `lameenc` package
(`pip install lameenc`) and the client needs `requests`.

## Notes

- XTTS inference is serialized with a lock, so the server handles one
  synthesis at a time. Run multiple replicas (one GPU each) to scale out.
- XTTS v2 weights are released under the **Coqui Public Model License (CPML)**,
  which is **non-commercial**. Review it before any production use.
- Set `TTS_DEVICE=cpu` to run without a GPU (much slower).
