import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from audio import load_audio, pad_or_trim, get_spectrogram
from model import Whisper, WhisperParams
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

params = WhisperParams()
whisper = Whisper(params)
whisper.load_weights()
whisper.float()

assert len(sys.argv) == 2, 'please provide audio file'
file = sys.argv[1]
test_filename = f'audio/{file}'

audio = load_audio(test_filename)
spec = get_spectrogram(torch.tensor(audio), params.num_mels)
spec = pad_or_trim(spec, length=params.audio_seq_len, axis=1)

sot_token = 50258 # <|startoftranscript|>
nts_token = 50364 # <|notimestamps|>
eot_token = 50257 # <|endoftext|>
ns_token = 50363 # <|nospeech|>
tokens = torch.tensor([sot_token, nts_token], dtype=torch.int64)

while True:
  out = whisper(spec.unsqueeze(0), tokens.unsqueeze(0))
  next_token = out.squeeze().argmax()
  print(f'next token: {next_token}')
  if next_token in [eot_token, ns_token]:
    break
  tokens = torch.cat((tokens, torch.tensor([next_token], dtype=tokens.dtype)), dim=0)
  tokens_list = tokens.cpu().tolist()
  print(f'text: {tokenizer.decode(tokens_list)}')
  print('\n')
