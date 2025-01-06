import torch
import torch.nn as nn
import torch.nn.functional as F
from audio import load_audio, pad_or_trim, get_spectrogram
from model import Whisper, WhisperParams
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

params = WhisperParams()
whisper = Whisper(params)
whisper.float()
print('loaded model')

test_filename = 'audio/english1.mp3'

audio = load_audio(test_filename)
spec = get_spectrogram(torch.tensor(audio))
spec = pad_or_trim(spec, length=params.audio_seq_len, axis=1)

tokens = torch.tensor([50258, 50364, 50257], dtype=torch.int64)
tokens = pad_or_trim(tokens, length=params.txt_seq_len, axis=0)

print('padded model inputs')
tokens = F.one_hot(tokens, num_classes=params.vocab_dim) # should be batch * seq length * embedding dim
out = whisper(spec.unsqueeze(0), tokens) # using blank tokens
print(out.shape)
