import torch
import torch.nn as nn
from audio import load_audio, pad_or_trim, get_spectrogram
from model import Whisper, WhisperParams
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
whisper = Whisper(WhisperParams())

test_filename = 'audio/english1.mp3'

audio = load_audio(test_filename)
spec = get_spectrogram(torch.tensor(audio))
spec = pad_or_trim(spec, length=1500, axis=1)
out = whisper(spec.unsqueeze(0), tokens=torch.tensor([50258, 50364, 50257], dtype=torch.int64)) # using blank tokens
print(out.shape)
