import os
import threading
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

print("Loading XTTS model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
# XTTS inference is not thread-safe, and FastAPI runs sync endpoints in a
# threadpool, so serialize access to the single shared model.
tts_lock = threading.Lock()
print("XTTS model loaded.")


class TTSRequest(BaseModel):
    text: str
    voice: str
    language: str = "pt"


@app.get("/health")
def health():
    return {"status": "ok"}


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

    with tts_lock:
        tts.tts_to_file(
            text=req.text,
            speaker_wav=str(voice),
            language=req.language,
            file_path=str(output),
        )

    # Stream the file back, then delete it so output/ doesn't grow unbounded.
    return FileResponse(
        path=output,
        media_type="audio/wav",
        filename=output.name,
        background=BackgroundTask(output.unlink, missing_ok=True),
    )
