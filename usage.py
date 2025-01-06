import torch
import torch.nn as nn
from audio import load_audio, pad_or_trim, get_spectrogram
from model import Whisper, WhisperParams
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
whisper = Whisper(WhisperParams())

audio = load_audio('english1.mp3')
spec = get_spectrogram(torch.tensor(audio))
spec = pad_or_trim(spec, length=1500, axis=1)
out = whisper(spec.unsqueeze(0), tokens=torch.tensor([50258, 50364, 50257], dtype=torch.float32)) # using blank tokens
print(out.shape)
