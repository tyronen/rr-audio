import logging
from contextlib import asynccontextmanager
import torch
from transformers import (
    AutoConfig,
    pipeline,
    Pipeline,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTimeStampLogitsProcessor,
)
from transformers.utils import is_flash_attn_2_available

from typing import Optional

import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
from starlette.websockets import WebSocket, WebSocketDisconnect
import utils
from faster_whisper import WhisperModel


@asynccontextmanager
async def lifespan(_: FastAPI):
    utils.setup_logging()
    _load_pipeline()
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

MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "openai/whisper-tiny")
DEVICE_PREFERENCE = utils.get_device().type

asr_pipe: Optional[Pipeline] = None
asr_model: Optional[WhisperModel] = None


def _load_pipeline() -> None:
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
    if "openai/whisper-tiny.en" in MODEL_NAME:
        config.alignment_heads = [
            [1, 0],
            [2, 0],
            [2, 5],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
        ]
    elif "openai/whisper-medium.en" in MODEL_NAME:
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

    # 1. Load your model object (so you can grab its generation_config)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # 2. Load your processor as before
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    # 3. Compute the “begin_index” as the number of forced_decoder_ids the model is already using:
    begin_index = len(model.generation_config.forced_decoder_ids or [])

    # 4. Instantiate the logits‐processor with the model’s GenerationConfig and that index:
    ts_processor = WhisperTimeStampLogitsProcessor(
        model.generation_config,
        begin_index,
        _detect_timestamp_from_logprob=True,  # you can omit or set False if you like
    )

    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        torch_dtype=torch.float16 if device_str == "cuda" else torch.float32,
        device=device_str,
        model_kwargs=attn_impl,
        chunk_length_s=ROLLING_WINDOW_SEC,
        stride_length_s=SLIDE_INTERVAL_SEC,
        return_timestamps="word",
        config=config,
        processor=processor,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        generate_kwargs={
            "logits_processor": [ts_processor],
            "num_beams": 4,
            "no_repeat_ngram_size": 3,
        },
    )

    Path("chunks").mkdir(exist_ok=True)


def _load_model() -> None:
    global asr_model

    if DEVICE_PREFERENCE.startswith("cuda"):
        device = "cuda"
        compute_type = "float16"
    elif DEVICE_PREFERENCE == "mps":
        device = "cpu"  # Apple Silicon: use CPU for now; float32 is default
        compute_type = "int8"  # Or "float32" if int8 fails
    else:
        device = "cpu"
        compute_type = "int8"

    model_name = MODEL_NAME.replace("openai/whisper-", "").replace(
        "distil-whisper/", ""
    )

    asr_model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )


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


def transcribe_pipeline(rolling_buffer, buffer_duration):
    global asr_pipe
    result = asr_pipe(rolling_buffer)
    text = result["text"].strip()
    chunks = result.get("chunks", [])
    tokens = [c["text"] for c in chunks]
    word_timestamps = [c["timestamp"] for c in chunks]
    # Detect incomplete timestamps and fall back if needed
    if any(len(ts) != 2 for ts in word_timestamps):
        logging.warning(
            "Incomplete timestamp detected; falling back to uniform timing."
        )
        raw = asr_pipe(rolling_buffer, return_timestamps=False)
        text = raw["text"].strip()
        tokens = text.split()
        word_timestamps = []
        for i in range(len(tokens)):
            s = i * buffer_duration / len(tokens)
            t = (i + 1) * buffer_duration / len(tokens)
            word_timestamps.append((s, t))
    return text, tokens, word_timestamps


def transcribe_model(rolling_buffer):
    segments, info = asr_model.transcribe(
        rolling_buffer,
        language="en",
        beam_size=4,
        word_timestamps=True,
        vad_filter=True,  # Optionally add VAD
    )

    tokens = []
    word_timestamps = []
    text = ""
    for segment in segments:
        text += segment.text
        for word in segment.words:
            tokens.append(word.word)
            word_timestamps.append((word.start, word.end))
    text = text.strip()
    return text, tokens, word_timestamps


def transcribe(buf, chunk_id, received_samples, buffer_duration):
    global asr_pipe, asr_model
    silence = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.int16)
    buf = np.concatenate((buf, silence))
    rolling_buffer = buf.astype(np.float32) / 32768.0
    end_sec = received_samples / SAMPLE_RATE
    start_sec = max(0.0, end_sec - buffer_duration)

    if asr_model:
        text, tokens, word_timestamps = transcribe_model(rolling_buffer)
    elif asr_pipe:
        text, tokens, word_timestamps = transcribe_pipeline(
            rolling_buffer, buffer_duration
        )
    else:
        raise Exception("Neither pipeline nor model configured")

    # Include tokens and timestamps in the JSON you send back
    logging.info(
        {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "chunk_id": chunk_id,
            "text": text,
        }
    )
    return {
        "chunk_id": chunk_id,
        "text": text,
        "tokens": tokens,
        "word_timestamps": word_timestamps,
        "start_sec": start_sec,
        "duration": buffer_duration,
    }


def transcribe_window(ring, chunk_id, received_samples):
    # Always use the last 30s (or less, if not enough)
    if ring.size >= rolling_window_samples:
        buf = ring[-rolling_window_samples:]
        buffer_duration = rolling_window_samples / SAMPLE_RATE
    else:
        buf = ring
        buffer_duration = ring.size / SAMPLE_RATE

    return transcribe(buf, chunk_id, received_samples, buffer_duration)


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
    if asr_pipe is None and asr_model is None:
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
        stream_ended = False
        while True:
            # Receive raw PCM bytes
            frame = await websocket.receive_bytes()
            if len(frame) == 0:
                stream_ended = True
                # fall through to flush logic below
            else:
                samples = np.frombuffer(frame, dtype=np.int16)
                # Append to ring-buffer
                ring = np.concatenate((ring, samples))
                received_samples += samples.size
                # Process every interval
                while received_samples >= next_emit_samples:
                    retval = transcribe_window(ring, chunk_id, received_samples)
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
            if stream_ended and ring.size > 0:
                # do one final pass on whatever is left
                buf = ring
                buffer_duration = buf.size / SAMPLE_RATE

                retval = transcribe(buf, chunk_id, received_samples, buffer_duration)
                await websocket.send_text(json.dumps(retval))
                # clear the ring so we don’t send again
                ring = np.empty((0,), dtype=np.int16)

            # Finally, once client stopped *and* ring is empty *and*
            # we’re not going to emit any more intervals, break and close
            if stream_ended and ring.size == 0 and received_samples < next_emit_samples:
                await websocket.close(code=1000, reason="flushed")
                return

    except WebSocketDisconnect as e:
        logging.error(f"WebSocket disconnected: {e}")
        # Client hung up – just exit the coroutine
        return


@app.post("/train")
async def train(audio: UploadFile = File(...), transcript: str = Form(...)):
    """
    Accepts an audio file and a corrected transcript. Fine-tunes the ASR model on this pair.
    """
    global asr_pipe

    # Only support insanely-fast-whisper for now (training requires PyTorch)
    if asr_pipe is None:
        raise HTTPException(
            status_code=503, detail="ASR model is not loaded or not trainable."
        )

    # Read the raw audio file into numpy
    audio_bytes = await audio.read()
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Prepare the model for training (single step)
    model = asr_pipe.model
    processor = asr_pipe.processor

    # Encode audio and target
    input_features = asr_pipe.feature_extractor(
        audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features.to(model.device)

    # Convert dtype if model expects float16 (half precision)
    if model.dtype == torch.float16:
        input_features = input_features.half()
    elif model.dtype == torch.bfloat16:
        input_features = input_features.bfloat16()
    else:
        input_features = input_features.float()

    labels = (
        processor(text=transcript, return_tensors="pt")
        .input_ids.to(model.device)
        .long()
    )

    # Standard training loop for a single step
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    outputs = model(input_features=input_features, labels=labels)
    loss = outputs.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Optionally, switch back to eval mode
    model.eval()

    return {"success": True, "loss": float(loss.detach().cpu().item())}
