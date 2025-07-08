import logging
import os
import json
import multiprocessing
from functools import partial
import utils
import torch
import torchaudio.transforms as T
import datasets
from tqdm import tqdm

from utils import SPECTROGRAM_DIR


def preprocess_audio(audio_array, sample_rate=22050, n_mels=64, max_duration=4.0):
    """Convert raw audio to mel spectrogram with fixed dimensions."""
    if not isinstance(audio_array, torch.Tensor):
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    else:
        audio_tensor = audio_array

    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.mean(dim=0)

    target_sr = 22_050
    if sample_rate != target_sr:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
        audio_tensor = resampler(audio_tensor)
        sample_rate = target_sr

    max_samples = int(max_duration * sample_rate)
    if len(audio_tensor) > max_samples:
        audio_tensor = audio_tensor[:max_samples]
    elif len(audio_tensor) < max_samples:
        padding = max_samples - len(audio_tensor)
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels, n_fft=2048, hop_length=512
    )
    mel_spec = mel_transform(audio_tensor)
    log_mel = torch.log(mel_spec + 1e-8)
    return log_mel.unsqueeze(0)


# A worker function to process and save one item.
def process_and_save(sample, index):
    """Processes a single audio sample and saves the spectrogram to disk."""
    # Extract metadata
    audio_array = sample['audio']['array']
    sample_rate = sample['audio']['sampling_rate']
    class_id = sample['classID']
    fold = sample['fold']

    # Create the spectrogram
    spectrogram = preprocess_audio(audio_array, sample_rate)

    # Define a unique path for the output file
    file_name = f"fold{fold}_class{class_id}_item{index}.pt"
    output_path = os.path.join(SPECTROGRAM_DIR, file_name)

    # Save the tensor to disk
    torch.save(spectrogram, output_path)

    # Return the metadata to be saved later
    return {"path": output_path, "class_id": class_id, "fold": fold}

def main():
    utils.setup_logging()
    """Loads data, preprocesses it in parallel, and saves results to disk."""
    os.makedirs(SPECTROGRAM_DIR, exist_ok=True)

    # Load the raw dataset
    raw_data = datasets.load_dataset("danavery/urbansound8K", split='train')

    # Use multiprocessing to speed up the preprocessing
    # os.cpu_count() uses all available CPU cores
    num_processes = os.cpu_count()
    logging.info(f"Starting preprocessing with {num_processes} processes...")

    # Create a partial function to pass the output_dir to the worker
    worker_func = partial(process_and_save)

    # Create a pool of workers
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use tqdm for a progress bar
        results = list(tqdm(pool.starmap(worker_func, enumerate(raw_data)), total=len(raw_data)))

    # Save the metadata for all processed files
    metadata_path = os.path.join(SPECTROGRAM_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info(f"Preprocessing complete. Spectrograms saved in '{SPECTROGRAM_DIR}'.")
    logging.info(f"Metadata saved to '{metadata_path}'.")

if __name__ == "__main__":
    main()
