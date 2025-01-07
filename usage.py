import torch
import torch.nn as nn
import torch.nn.functional as F
from audio import load_audio, pad_or_trim, get_spectrogram
from model import Whisper, WhisperParams
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

params = WhisperParams()
whisper = Whisper(params)
whisper.load_weights()
whisper.float()

test_filename = 'audio/english1.mp3'

audio = load_audio(test_filename, params.num_mels)
spec = get_spectrogram(torch.tensor(audio))
spec = pad_or_trim(spec, length=params.audio_seq_len, axis=1)

sot_token = 50258 # <|startoftranscript|>
nts_token = 50364 # <|notimestamps|>
eot_token = 50257 # <|endoftext|>
tokens = torch.tensor([sot_token, nts_token], dtype=torch.int64)

while True:
  out = whisper(spec.unsqueeze(0), tokens.unsqueeze(0))
  next_token = out.squeeze().argmax()
  print(f'next token: {next_token}')
  if next_token == eot_token:
    break
  tokens = torch.cat((tokens, torch.tensor([next_token], dtype=tokens.dtype)), dim=0)
  tokens_list = tokens.cpu().tolist()
  print(f'text: {tokenizer.decode(tokens_list)}')
  print('\n')
