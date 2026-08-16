import os
import queue
from pathlib import Path
from uuid import uuid4

# XTTS v2 is distributed under the Coqui Public Model License. On first run the
# model download prompts for agreement on stdin, which would hang a headless
# container, so agree non-interactively before importing TTS.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

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

# XTTS is autoregressive: a single stream leaves most of the GPU idle (~25% on a
# 16 GB card). Holding several independent model instances and serving requests
# concurrently fills those gaps. Each extra instance costs ~3 GB of VRAM (the
# CUDA context is shared within the process), so 3 fits comfortably in 16 GB.
POOL_SIZE = int(os.environ.get("TTS_POOL_SIZE", "3"))

# XTTS inference is not thread-safe, so a model is checked out of the pool for
# the duration of a request and returned afterwards. The queue doubles as the
# admission control: request N+1 blocks until an instance frees up.
print(f"Loading {POOL_SIZE} XTTS model instance(s)...")
tts_pool: queue.Queue = queue.Queue()
for i in range(POOL_SIZE):
    tts_pool.put(TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE))
    print(f"  instance {i + 1}/{POOL_SIZE} loaded.")
print("XTTS models loaded.")


class TTSRequest(BaseModel):
    text: str
    voice: str
    language: str = "pt"


@app.get("/health")
def health():
    return {"status": "ok", "pool_size": POOL_SIZE, "idle": tts_pool.qsize()}


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

    tts = tts_pool.get()
    try:
        tts.tts_to_file(
            text=req.text,
            speaker_wav=str(voice),
            language=req.language,
            file_path=str(output),
        )
    finally:
        tts_pool.put(tts)

    # Stream the file back, then delete it so output/ doesn't grow unbounded.
    return FileResponse(
        path=output,
        media_type="audio/wav",
        filename=output.name,
        background=BackgroundTask(output.unlink, missing_ok=True),
    )
