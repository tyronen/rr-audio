import logging
from contextlib import asynccontextmanager
import torch
from transformers import AutoConfig, pipeline, Pipeline
from transformers.utils import is_flash_attn_2_available

from typing import List, Optional

import os
from pathlib import Path

from fastapi import FastAPI
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


# Model guide:
# openai/whisper-tiny: 39m
# openai/whisper-base: 74m
# distil-whisper/distil-small.en: 166m, short-form wer 12.1
# openai/whisper-small: 244m
# distil-whisper/distil-medium.en: 394m, 11.1
# distil-whisper/distil-large-v3: 756m, 9.7
# openai/whisper-medium: 769m
# openai/whisper-large-v3-turbo: 809m
# openai/whisper-large-v3: 1550m, 8.4

MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "openai/whisper-medium.en")
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

    config = AutoConfig.from_pretrained(MODEL_NAME)
    # Set alignment_heads for specific models
    if "medium.en" in MODEL_NAME:
        # alignment heads for medium.en
        config.alignment_heads = [
            [11, 4],
            [14, 1],
            [14, 12],
            [14, 14],
            [15, 4],
            [16, 0],
            [16, 4],
            [16, 9],
            [17, 12],
            [17, 14],
            [18, 7],
            [18, 10],
            [18, 15],
            [20, 0],
            [20, 3],
            [20, 9],
            [20, 14],
            [21, 12],
        ]
    elif not getattr(config, "alignment_heads", None):
        # default fallback
        config.alignment_heads = [[0, 0]]

    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        torch_dtype=torch.float16 if device_str != "cpu" else torch.float32,
        device=device_str,
        model_kwargs=attn_impl,
        chunk_length_s=ROLLING_WINDOW_SEC,
        return_timestamps="word",
        config=config,
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

# Initial exponential send intervals (seconds): then full window
INITIAL_SEND_INTERVALS = [2, 5, 10, 20, ROLLING_WINDOW_SEC]
# After initial intervals, slide window
SLIDE_INTERVAL_SEC = ROLLING_WINDOW_SEC / 3

rolling_window_samples = ROLLING_WINDOW_SEC * SAMPLE_RATE


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
    chunk_id = 0

    # Emission scheduling state
    emit_index = 0
    next_emit_sec = INITIAL_SEND_INTERVALS[emit_index]
    next_emit_samples = int(next_emit_sec * SAMPLE_RATE)

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

            # Process every interval
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

                try:
                    result = asr_pipe(rolling_buffer)
                    text = result["text"].strip()
                    chunks = result.get("chunks", [])
                    tokens = [c["text"] for c in chunks]
                    word_timestamps = [c["timestamp"] for c in chunks]
                except (ValueError, IndexError) as e:
                    logging.warning(
                        f"Timestamps error ({type(e).__name__}): {e}. Falling back to uniform timing."
                    )
                    raw = asr_pipe(rolling_buffer, return_timestamps=False)
                    text = raw["text"].strip()
                    tokens = text.split()
                    # build uniform [start, end] for each token
                    word_timestamps = []
                    for i in range(len(tokens)):
                        s = i * buffer_duration / len(tokens)
                        t = (i + 1) * buffer_duration / len(tokens)
                        word_timestamps.append((s, t))
                # Include tokens and timestamps in the JSON you send back
                retval = {
                    "chunk_id": chunk_id,
                    "text": text,
                    "tokens": tokens,
                    "word_timestamps": word_timestamps,
                    "start_sec": start_sec,
                    "duration": buffer_duration,
                }

                logging.info({"chunk_id": chunk_id, "text": text})
                await websocket.send_text(json.dumps(retval))
                chunk_id += 1

                # Advance to next emit threshold
                emit_index += 1
                if emit_index < len(INITIAL_SEND_INTERVALS):
                    next_emit_sec = INITIAL_SEND_INTERVALS[emit_index]
                else:
                    # after initial, slide by half-window increments
                    next_emit_sec = ROLLING_WINDOW_SEC + SLIDE_INTERVAL_SEC * (
                        emit_index - len(INITIAL_SEND_INTERVALS) + 1
                    )
                next_emit_samples = int(next_emit_sec * SAMPLE_RATE)
            if ring.size > rolling_window_samples:
                ring = ring[-rolling_window_samples:]

    except WebSocketDisconnect as e:
        logging.error(f"WebSocket disconnected: {e}")
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
