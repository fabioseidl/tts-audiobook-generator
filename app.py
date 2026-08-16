import os
import queue
import wave
from pathlib import Path
from uuid import uuid4

# XTTS v2 is distributed under the Coqui Public Model License. On first run the
# model download prompts for agreement on stdin, which would hang a headless
# container, so agree non-interactively before importing TTS.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from TTS.api import TTS

app = FastAPI(title="XTTS Server")

VOICE_DIR = Path("voices")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = os.environ.get("TTS_DEVICE", "cuda")

# Keep this at 1 and scale with uvicorn workers (WEB_CONCURRENCY) instead.
# XTTS decoding is an autoregressive Python loop that holds the GIL, so extra
# instances inside one interpreter take turns rather than overlapping: measured
# at pool size 5, three concurrent requests used 181% CPU and left the GPU at
# 13%. Separate worker processes each get their own GIL and actually overlap.
POOL_SIZE = int(os.environ.get("TTS_POOL_SIZE", "1"))


def write_wav(path: Path, samples, sample_rate: int) -> None:
    """Write float samples in [-1, 1] as a 16-bit mono WAV."""
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((pcm * 32767).astype("<i2").tobytes())


class Engine:
    """One XTTS model instance plus its cached speaker conditioning latents."""

    def __init__(self, device: str):
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.model = tts.synthesizer.tts_model
        self.sample_rate = tts.synthesizer.output_sample_rate
        self._latents: dict[str, tuple] = {}

    def conditioning(self, voice: Path) -> tuple:
        """Return (gpt_cond_latent, speaker_embedding) for a reference voice.

        ``tts_to_file(speaker_wav=...)`` recomputes these on every call, which
        means decoding and resampling the whole reference WAV to synthesize a
        200-character part. The latents depend only on the reference audio, so
        compute them once per voice and keep them.
        """
        key = voice.name
        if key not in self._latents:
            print(f"Computing conditioning latents for {key}...")
            self._latents[key] = self.model.get_conditioning_latents(
                audio_path=[str(voice)]
            )
        return self._latents[key]

    def synthesize(self, text: str, voice: Path, language: str):
        gpt_cond_latent, speaker_embedding = self.conditioning(voice)
        return self.model.inference(
            text, language, gpt_cond_latent, speaker_embedding
        )["wav"]


# XTTS inference is not thread-safe, and FastAPI runs sync endpoints in a
# threadpool, so an engine is checked out for the duration of a request and
# returned afterwards. The queue doubles as admission control: with POOL_SIZE=1
# a second request to the same worker waits instead of corrupting state.
print(f"Loading {POOL_SIZE} XTTS model instance(s)...")
engine_pool: queue.Queue = queue.Queue()
for i in range(POOL_SIZE):
    engine_pool.put(Engine(DEVICE))
    print(f"  instance {i + 1}/{POOL_SIZE} loaded.")
print("XTTS models loaded.")


class TTSRequest(BaseModel):
    text: str
    voice: str
    language: str = "pt"


@app.get("/health")
def health():
    return {"status": "ok", "pool_size": POOL_SIZE, "idle": engine_pool.qsize()}


@app.get("/voices")
def voices():
    return {"voices": sorted(p.name for p in VOICE_DIR.glob("*.wav"))}


@app.post("/tts")
def generate(req: TTSRequest):
    # Path(...).name strips any directory components to prevent path traversal
    # (e.g. voice="../../etc/passwd").
    voice = VOICE_DIR / Path(req.voice).name
    if not voice.exists():
        raise HTTPException(status_code=404, detail="Voice not found")

    output = OUTPUT_DIR / f"{uuid4()}.wav"

    engine = engine_pool.get()
    try:
        samples = engine.synthesize(req.text, voice, req.language)
        write_wav(output, samples, engine.sample_rate)
    finally:
        engine_pool.put(engine)

    # Stream the file back, then delete it so output/ doesn't grow unbounded.
    return FileResponse(
        path=output,
        media_type="audio/wav",
        filename=output.name,
        background=BackgroundTask(output.unlink, missing_ok=True),
    )
