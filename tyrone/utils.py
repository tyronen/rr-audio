import logging
from contextlib import nullcontext

import torch
from torch.cuda.amp import autocast, GradScaler


START_TOKEN = 10  # After digits 0-9
END_TOKEN = 11
BLANK_TOKEN = 12


SIMPLE_MODEL_FILE = "data/simple.pth"
COMPLEX_MODEL_FILE = "data/complex.pth"


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
        return autocast(), GradScaler()
    else:
        # fall-back: no automatic casting, dummy scaler
        return nullcontext(), GradScaler(enabled=False)
