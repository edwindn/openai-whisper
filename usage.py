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

test_filename = 'audio/english1.mp3'

audio = load_audio(test_filename)
spec = get_spectrogram(torch.tensor(audio))
spec = pad_or_trim(spec, length=params.audio_seq_len, axis=1)

tokens = torch.tensor([50258, 50364, 50257], dtype=torch.int64).unsqueeze(0)

while True:
  out = whisper(spec.unsqueeze(0), tokens) # using blank tokens
  print(out.shape)
  next_token = out.squeeze().argmax()
  print(f'next token: {next_token}')
  tokens = torch.cat((tokens, next_token.to(tokens.dtype)), dim=0)
  tokens_list = tokens.cpu().tolist()
  print(f'text: {tokenizer.decode(tokens_list)}')
  print('\n')
