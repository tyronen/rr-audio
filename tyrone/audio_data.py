import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset


def preprocess_audio(audio_array, sample_rate=22050, n_mels=64, max_duration=4.0):
    """Convert raw audio to mel spectrogram with fixed dimensions."""
    # Convert to tensor if needed
    if not isinstance(audio_array, torch.Tensor):
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    else:
        audio_tensor = audio_array

    # Ensure mono
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.mean(dim=0)

    # ------------------------------------------------------------------
    # Resample to a fixed target rate so that every spectrogram
    # ends up with the same time dimension (avoids DataLoader stack errors)
    target_sr = 22_050
    if sample_rate != target_sr:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
        audio_tensor = resampler(audio_tensor)
        sample_rate = target_sr  # keep downstream logic consistent
    # ------------------------------------------------------------------

    # First, fix the audio length to ensure consistent spectrogram size
    max_samples = int(max_duration * sample_rate)
    if len(audio_tensor) > max_samples:
        # Trim to max_samples
        audio_tensor = audio_tensor[:max_samples]
    elif len(audio_tensor) < max_samples:
        # Pad to max_samples
        padding = max_samples - len(audio_tensor)
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))

    # Convert to mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        f_min=20.0,
        f_max=8000.0,
        power=2.0,
        normalized=False,
    )

    mel_spec = mel_transform(audio_tensor)
    
    # Convert to log scale and add channel dimension
    log_mel = torch.log(mel_spec + 1e-8)
    return log_mel.unsqueeze(0)  # Add channel dimension for CNN


class UrbanSoundDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        sample = self.dataset[idx]

        # Get audio and class
        audio = sample['audio']['array']
        sample_rate = sample['audio']['sampling_rate']
        class_id = sample['classID']

        # Preprocess audio to spectrogram
        spectrogram = preprocess_audio(audio, sample_rate)

        item = (spectrogram, class_id)
        self.cache[idx] = item
        return item
