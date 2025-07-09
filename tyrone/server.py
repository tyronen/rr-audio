from contextlib import asynccontextmanager

import torch
from transformers import pipeline, Pipeline
from transformers.utils import is_flash_attn_2_available

from typing import List, Optional

import asyncio
import os
import subprocess
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import numpy as np  # pip install numpy
from starlette.websockets import WebSocket, WebSocketDisconnect


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------------------------------
# Whisper model (loaded once at startup)
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "distil-whisper/distil-small.en")
DEVICE_PREFERENCE = os.getenv("WHISPER_DEVICE", "mps")  # "cuda:0", "mps", or "cpu"

asr_pipe: Optional[Pipeline] = None


def _load_model() -> None:
    """
    Load the insanely‑fast‑whisper checkpoint via Hugging Face Transformers.
    Uses Flash‑Attention‑2 if available, otherwise SDPA attention.
    """
    global asr_pipe

    # Decide device string
    if DEVICE_PREFERENCE.startswith("cuda") and torch.cuda.is_available():
        device_str = DEVICE_PREFERENCE
    elif DEVICE_PREFERENCE == "mps" and torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"

    attn_impl = (
        {"attn_implementation": "flash_attention_2"}
        if is_flash_attn_2_available()
        else {"attn_implementation": "sdpa"}
    )

    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        torch_dtype=torch.float16 if device_str != "cpu" else torch.float32,
        device=device_str,
        model_kwargs=attn_impl,
    )

    Path("chunks").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TranscribeResponse(BaseModel):
    chunk_id: int
    text: str
    tokens: List[str]
    word_timestamps: List[float]


class FeedbackRequest(BaseModel):
    chunk_id: int
    corrected_text: str


# ---------------------------------------------------------------------------
# End‑points
# ---------------------------------------------------------------------------


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    chunk_id: int = Form(...),
    audio: UploadFile = File(...),
) -> TranscribeResponse:
    """
    Accepts a 30‑second WebM/Opus audio chunk plus its numeric `chunk_id`
    from the front‑end.  Saves the file and kicks off ASR (to be implemented).

    TODO:
      • Write the uploaded audio to `./chunks/{chunk_id}.webm`
      • Run insanely-faster-whisper and fill `text`, `tokens`,
        `word_timestamps`.
    """
    # -----------------------------------------------------------------------
    # 1.  Persist the uploaded blob to disk
    # -----------------------------------------------------------------------
    chunk_dir = Path("chunks")
    chunk_dir.mkdir(exist_ok=True)

    payload = await audio.read()

    # Skip blobs that are obviously too small (<4 kB) – they have no header.
    if len(payload) < 4_096:
        raise HTTPException(status_code=400, detail="Empty or too‑small audio blob")

    mime = (audio.content_type or "").lower()
    ext = ".ogg" if "ogg" in mime else ".webm"

    chunk_path = chunk_dir / f"{chunk_id}{ext}"
    with chunk_path.open("wb") as f:
        f.write(payload)

    # Convert WebM/Opus ➜ 16‑kHz mono WAV (Whisper default)
    wav_path = chunk_path.with_suffix(".wav")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",  # overwrite
            "-probesize",
            "1M",
            "-analyzeduration",
            "1M",
            "-i",
            str(chunk_path),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or "Unknown ffmpeg failure"
        print(f"[ffmpeg] {err_msg}")  # log to server console
        raise HTTPException(
            status_code=400,
            detail=f"FFmpeg failed: {err_msg}",
        )

    # -----------------------------------------------------------------------
    # 2.  Run Whisper inference in a worker thread (sync function off‑loaded)
    # -----------------------------------------------------------------------
    def _do_transcribe_sync() -> TranscribeResponse:
        global asr_pipe
        if asr_pipe is None:
            raise RuntimeError("ASR pipeline not initialised")

        result = asr_pipe(
            str(wav_path),
            chunk_length_s=10,
            batch_size=1,
            return_timestamps=True,
        )

        tokens: list[str] = []
        word_ts: list[float] = []

        for ch in result.get("chunks", []):
            tokens.append(ch["text"])
            if ch["timestamp"] is not None:
                word_ts.append(ch["timestamp"][0])

        return TranscribeResponse(
            chunk_id=chunk_id,
            text=result["text"].strip(),
            tokens=tokens,
            word_timestamps=word_ts,
        )

    return await asyncio.to_thread(_do_transcribe_sync)


# ---------------------------------------------------------------------------
# WebSocket endpoint for raw 16-kHz PCM streaming
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_pcm(websocket: WebSocket):
    """
    Client sends binary Int16 PCM frames (little-endian, 16 kHz, mono).
    We keep a simple ring-buffer; every 32 000 samples (~2 s) we run the
    Whisper pipeline and push the text back.

    Outgoing message shape:
        {"chunk_id": <int>, "text": "<transcript>"}
    """
    await websocket.accept()

    # Make sure the ASR model is loaded
    global asr_pipe
    if asr_pipe is None:
        await websocket.close(code=1011, reason="ASR model not ready")
        return

    WINDOW = 32_000  # 2 s × 16 kHz = 32 000 samples
    ring = np.empty((0,), dtype=np.int16)  # rolling PCM buffer
    chunk_id = 0

    try:
        while True:
            # Receive raw PCM bytes
            frame = await websocket.receive_bytes()
            samples = np.frombuffer(frame, dtype=np.int16)
            if samples.size == 0:
                continue

            # Append to ring-buffer
            ring = np.concatenate((ring, samples))

            # ßProcess every full 2-second window
            while ring.size >= WINDOW:
                window = ring[:WINDOW].astype(np.float32) / 32768.0  # [-1, 1]
                ring = ring[WINDOW:]  # pop

                result = asr_pipe(
                    window,
                    return_timestamps=False,
                )
                text = result["text"].strip()

                await websocket.send_text(
                    json.dumps({"chunk_id": chunk_id, "text": text})
                )
                chunk_id += 1

    except WebSocketDisconnect:
        # Client hung up – just exit the coroutine
        return


@app.post("/feedback")
async def feedback(data: FeedbackRequest):
    """
    Receives user‑corrected transcription for a previously processed chunk.

    TODO:
      • Store `data.corrected_text` next to the original chunk
      • Diff original vs corrected and append to adaptation queue.
    """
    return {"status": "received", "chunk_id": data.chunk_id}
