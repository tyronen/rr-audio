import logging
from contextlib import nullcontext
import torch
from torch.amp import autocast, GradScaler

SPECTROGRAM_DIR = "/tmp/processed_spectrograms"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def amp_components(device, train=False):
    if device.type == "cuda" and train:
        return autocast(device_type="cuda"), GradScaler()
    else:
        # fall-back: no automatic casting, dummy scaler
        return nullcontext(), GradScaler(enabled=False)
