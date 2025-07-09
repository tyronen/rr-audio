import logging
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
import utils


@asynccontextmanager
async def lifespan(_: FastAPI):
    utils.setup_logging()
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

MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "openai/whisper-small")
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
# WebSocket endpoint for raw 16-kHz PCM streaming
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
ROLLING_WINDOW_SEC = 30
SEND_INTERVAL_SEC = 2

rolling_window_samples = ROLLING_WINDOW_SEC * SAMPLE_RATE
send_interval_samples = SEND_INTERVAL_SEC * SAMPLE_RATE


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

    ring = np.empty((0,), dtype=np.int16)  # rolling PCM buffer
    received_samples = 0
    next_emit_samples = send_interval_samples
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
            received_samples += samples.size

            # Process every 2s
            while received_samples >= next_emit_samples:
                # Always use the last 30s (or less, if not enough)
                if ring.size >= rolling_window_samples:
                    buf = ring[-rolling_window_samples:]
                    buffer_duration = rolling_window_samples / SAMPLE_RATE
                else:
                    buf = ring
                    buffer_duration = ring.size / SAMPLE_RATE

                rolling_buffer = buf.astype(np.float32) / 32768.0
                end_sec = received_samples / SAMPLE_RATE
                start_sec = max(0.0, end_sec - buffer_duration)
                logging.info(
                    f"size {rolling_buffer.size} duration {buffer_duration} "
                    f"received_samples {received_samples} end_sec {end_sec} start_sec {start_sec}"
                )

                result = asr_pipe(
                    rolling_buffer,
                    return_timestamps=False,
                )
                text = result["text"].strip()

                retval = {
                    "chunk_id": chunk_id,
                    "text": text,
                    "start_sec": start_sec,
                    "duration": buffer_duration,
                }
                logging.info(retval)
                await websocket.send_text(json.dumps(retval))
                chunk_id += 1
                next_emit_samples += send_interval_samples
            if ring.size > rolling_window_samples:
                ring = ring[-rolling_window_samples:]

    except Exception:
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
