import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from subprocess import CalledProcessError, run
SAMPLE_RATE = 16000
FFT_LENGTH = 400
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_audio(file, sr=SAMPLE_RATE):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]

    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    
"""
def load_audio(file):
    try:
            out, _ = (
                ffmpeg
                #.input(file)
                .input(file, threads=0, loglevel="quiet", ar=SAMPLE_RATE)
                .output('pipe:', format='wav')
                .run(capture_stdout=True, capture_stderr=True)
            )
    except ffmpeg.Error as e:
        print(f"Error while reading audio file {file}: {e.stderr}")
        return None
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
"""

def pad_or_trim(audio, length, axis=-1):
    assert torch.is_tensor(audio), "Can't pad or trim a non-tensor array"
    if audio.shape[axis] > length:
        return audio.index_select(dim=axis, index=torch.arange(length))
    elif audio.shape[axis] < length:
        padding = [(0, 0)] * audio.ndim
        padding[axis] = (0, length - audio.shape[axis])
        return F.pad(audio, [pad for sizes in padding[::-1] for pad in sizes])

def mel_filters(num_mels, file = 'mel_filters.npz'):
    with np.load(file, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{num_mels}"]).to(device)

def get_spectrogram(audio, num_mels):
    hop_time = 10e-3
    hop = int(SAMPLE_RATE * hop_time)
    window = torch.hann_window(FFT_LENGTH).to(device)
    ft_audio = torch.stft(audio, FFT_LENGTH, hop, window=window, return_complex=True)
    amplitues = ft_audio[...,:-1].abs()**2 # ?

    filters = mel_filters(num_mels)
    spectrogram = filters @ amplitues
    log_spectrogram = torch.clamp(spectrogram, 1e-10).log10()
    #log_spectrogram = torch.maximum(log_spectrogram, log_spectrogram.max() - 8)
    #log_spectrogram = (log_spectrogram + 4.) / 4.
    return log_spectrogram 
    
