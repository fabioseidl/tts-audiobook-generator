# XTTS Server

A FastAPI service that wraps [XTTS v2](https://huggingface.co/coqui/XTTS-v2)
for text-to-speech with voice cloning, plus a client that turns a markdown
ebook into an audiobook. Runs on GPU inside Docker.

Built on the maintained [`coqui-tts`](https://pypi.org/project/coqui-tts/)
package (the idiap fork of the original, now-abandoned Coqui `TTS`), which
supports Python 3.12 and current PyTorch.

## Requirements

- NVIDIA GPU with a recent driver, 12 GB VRAM or more for the default 3 workers
- Docker + the NVIDIA Container Toolkit (so `--gpus` / `deploy.devices` works)
- Python 3.10+ on the host for the audiobook client (`requests`, `lameenc`)

## Layout

```
voices/   reference speaker .wav clips (bind-mounted, read at request time)
output/   generated audio (bind-mounted, files auto-deleted after download)
models/   XTTS model cache (bind-mounted, persisted across restarts)
```

Create the folders and drop at least one reference clip in `voices/` before
starting. A clean 6–20 s mono WAV of the target speaker works well; anything
longer or at a higher sample rate is truncated to 30 s and resampled to
22.05 kHz mono internally.

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

## Configuration

| Variable          | Default | Description                                        |
|-------------------|---------|----------------------------------------------------|
| `WEB_CONCURRENCY` | `3`     | uvicorn worker processes; each loads its own model  |
| `TTS_POOL_SIZE`   | `1`     | model instances per worker                          |
| `TTS_DEVICE`      | `cuda`  | set to `cpu` to run without a GPU (much slower)     |

Concurrency comes from processes, not threads. XTTS decodes autoregressively in
a Python loop that holds the GIL, so multiple model instances inside a single
interpreter take turns rather than overlapping. Scale with `WEB_CONCURRENCY`
and leave `TTS_POOL_SIZE` at `1`; the in-process pool exists only to keep a
worker from running two synthesis calls at once, which XTTS does not support.

Each worker holds a full copy of the model: ~3.5 GB VRAM and ~3 GB host RAM.
Three workers occupy ~10.9 GB VRAM. Startup loads every worker's model before
the service accepts traffic, which is why the healthcheck allows 600 s.

## API

| Method | Path      | Description                                                   |
|--------|-----------|---------------------------------------------------------------|
| GET    | `/health` | Liveness; reports the answering worker's `pool_size` and `idle` |
| GET    | `/voices` | List available reference voices                                |
| POST   | `/tts`    | Synthesize speech, returns a WAV file                          |

```bash
curl -X POST http://127.0.0.1:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Olá, isto é um teste.", "voice": "my_speaker.wav", "language": "pt"}' \
  --output result.wav
```

`voice` is a filename in `voices/`; directory components are stripped. `text`
is synthesized as a single sequence. `language` is a supported XTTS code (`en`,
`pt`, `es`, `fr`, `de`, `it`, ...) and defaults to `pt`. The response is 16-bit
mono WAV at 24 kHz.

The first request for a given voice computes that voice's speaker conditioning
latents (~2 s) and caches them for the life of the worker. The cache is per
worker process, so the cost is paid once per worker.

### Connect over `127.0.0.1`, not `localhost`

Docker publishes the port on both `0.0.0.0:8000` and `[::]:8000`, but on
Windows the IPv6 path accepts the connection and then hangs. `localhost`
resolves to `::1` first, so every new connection stalls ~21 s before falling
back to IPv4 — 21,049 ms per request versus 3 ms on `127.0.0.1`. Clients should
also reuse connections rather than opening one per request.

## Audiobook generator (`tools/`)

[`tools/generate_audiobook.py`](tools/generate_audiobook.py) drives the TTS
server to convert a whole ebook. It reads a markdown file, strips the markdown
syntax, splits the text into parts on sentence/clause boundaries
(`<= --max-chars`, default 200), and POSTs each part to the API, saving one WAV
per part. Parts are synthesized `--workers` at a time over a shared keep-alive
connection pool.

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

Every part's audio is validated as a readable WAV and retried `--retries`
times. An unrecoverable failure cancels the queued parts, lets in-flight parts
finish, writes `parts.json`, and prints the exact
`--start-id N --skip-existing` command to resume with.

Start the TTS server first, then:

```bash
python tools/generate_audiobook.py                 # full run
python tools/generate_audiobook.py --dry-run       # only build parts.json, no synthesis
python tools/generate_audiobook.py --limit 5       # only the first 5 parts
python tools/generate_audiobook.py --start-id 42   # resume from part 42
python tools/generate_audiobook.py --skip-existing # skip parts already synthesized
python tools/generate_audiobook.py --workers 1     # strictly serial
python tools/generate_audiobook.py --join          # concatenate WAVs into grouped MP3s
```

Set `--workers` equal to the server's `WEB_CONCURRENCY` (both default to 3).
Higher only queues requests; lower leaves workers idle.

Other options: `--input`, `--output`, `--url`, `--voice`, `--language`,
`--max-chars`, `--timeout`, `--retries`, `--retry-delay`.

`--join` is a separate post-processing step that concatenates the per-part WAVs
into MP3 files under `output/audiobook/`, `--group-size` parts each (default
1000, so parts 1–1000 become `audiobook_part_1.mp3`, and so on) at `--bitrate`
kbps. It requires the `lameenc` package and does not contact the server.

## Performance

Measured on an RTX 5080 (16 GB), Portuguese text, Docker Desktop on Windows,
`WEB_CONCURRENCY=3` and `--workers 3`:

| Metric                              | Value              |
|-------------------------------------|--------------------|
| Single synthesis call, warm         | ~6.2 s             |
| Throughput, 3 workers               | ~3.6 s per part    |
| Audio produced per part (200 chars) | ~9 s               |
| Ratio to real time                  | ~0.4x              |
| GPU utilization                     | ~24%               |
| VRAM                                | ~10.9 GB           |

A 13,000-part book takes roughly 13 hours.

GPU utilization stays low because XTTS decodes autoregressively through many
small kernels and cannot saturate a modern GPU regardless of how the server is
arranged. Optimize wall-clock throughput, not utilization. The practical limits
are VRAM (each worker needs ~3.5 GB) and per-request Python overhead.

## Notes

- XTTS v2 weights are released under the **Coqui Public Model License (CPML)**,
  which is **non-commercial**. Review it before any production use.
- XTTS inference is not thread-safe. FastAPI runs sync endpoints in a
  threadpool, so each concurrent request checks out its own model instance from
  the worker's pool; a request arriving with none free waits for one.
- Text is synthesized without sentence splitting, so a part's length directly
  drives generation time. `--max-chars` controls it.
