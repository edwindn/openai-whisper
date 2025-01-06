import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

def sinusoidal_encoding(seq_len, dim, max_timescale=10000):
    PE = np.empty((seq_len, dim))
    pos = np.arange(seq_len).reshape(-1, 1)
    i = np.arange(dim).reshape(1, -1)
    inv = max_timescale ** (2/dim * i//2)
    PE[:,::2] = np.sin(pos / inv[:,::2])
    PE[:,1::2] = np.cos(pos / inv[:,1::2])
    return torch.tensor(PE)

class MHA(nn.Module): ### implement kv_cache for speeding up cross attention in decoding stage
    def __init__(self, input_dim, num_heads):
        super().__init__()
        assert input_dim % num_heads == 0, 'Input dimension should be divisble by number of attention heads'
        self.input_dim = input_dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x, xa=None, kv_cache=None, mask=None): # xa is for cross-attention
                                                             # xa needs same dim as x ?
        q = self.q_proj(x)

        batch, seq_len, _ = x.size()
        q = q.view(batch, seq_len, self.num_heads, -1).permute(0, 2, 1, 3) # batch, heads, sequence len, embedding

        if kv_cache is None:
            k = self.k_proj(x if xa is None else xa)
            v = self.v_proj(x if xa is None else xa)
        else:
            k = kv_cache[self.k_proj]
            v = kv_cache[self.v_proj]

        k = k.view(batch, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.view(batch, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)

        value = scaled_dot_product_attention(q, k, v, mask)
        value = value.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.out_proj(value)


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, cross_attention=False):
        super().__init__()

        self.self_attn = MHA(input_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(input_dim)

        if cross_attention:
            self.cross_attn = cross_attention
            self.encoder_attn = MHA(input_dim, num_heads) # should be named decoder_attn, but HF weights use encoder_attn
            self.encoder_attn_layer_norm = nn.LayerNorm(input_dim)

        self.fc1 = nn.Linear(input_dim, 4*input_dim)
        self.fc2 = nn.Linear(4*input_dim, input_dim)
        self.final_layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x, xa=None, mask=None): # IMPLEMENT KV CACHE
        x = x + self.self_attn(self.self_attn_layer_norm(x), mask)

        if self.cross_attn:
            assert xa is not None, "Audio encoding missing from cross-attention"
            x = x + self.encoder_attn(self.encoder_attn_layer_norm(x), xa, mask)

        x = self.final_layer_norm(x)
        x = x + self.fc2(F.gelu(self.fc1(x)))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, num_mels, input_dim, num_heads, seq_len, num_blocks):
        super().__init__()

        self.layers = nn.ModuleList([TransformerBlock(input_dim, num_heads) for _ in range(num_blocks)])

        self.conv1 = nn.Conv1d(num_mels, input_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.layer_norm = nn.LayerNorm(input_dim)

        #self.register_buffer("embed_positions", sinusoidal_encoding(seq_len, input_dim))
        self.embed_positions = sinusoidal_encoding(seq_len, input_dim)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1) # batch size, sequence length, input dim

        x = x + self.embed_positions

        for l in self.layers:
            x = l(x)

        x = self.layer_norm(x)
        return x


class TextDecoder(nn.Module):
    def __init__(self, vocab_dim, seq_len, input_dim, num_heads, num_blocks):
        super().__init__()

        self.embed_tokens = nn.Embedding(vocab_dim, input_dim)
        self.embed_positions = nn.Embedding(seq_len, input_dim)
        self.layers = nn.ModuleList([TransformerBlock(input_dim, num_heads, cross_attention=True) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(input_dim)

        mask = torch.empty(seq_len, seq_len).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, xa): # xa is the audio encoding, x are the existing output text tokens
                              # seq_len is fixed - need to pad

        x = self.embed_tokens(x) + self.embed_positions
        for l in self.layers:
            x = l(x, xa, mask=self.mask)

        x = self.layer_norm(x) # same shape as positional embedding
        logits = x @ torch.transpose(self.embed_tokens.weight.to(x.dtype), 0, 1) # input_dim -> vocab_dim
        return logits


class Whisper(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.encoder = AudioEncoder(
            params.num_mels,
            params.audio_input_dim,
            params.audio_num_heads,
            params.audio_seq_len,
            params.audio_num_blocks
        )
        self.decoder = TextDecoder(
            params.vocab_dim,
            params.txt_seq_len,
            params.txt_input_dim,
            params.txt_num_heads,
            params.txt_num_blocks
        )

    def forward(self, mel, tokens): # takes in existing output text tokens
        return self.decoder(tokens, self.encoder(mel)) # logits for next token prediction

    def embed_audio(self, mel):
        return self.encoder(mel)


from dataclasses import dataclass
@dataclass
class WhisperParams:
    num_mels: int = 80
    audio_input_dim: int = 1280
    audio_num_heads: int = 4
    audio_seq_len: int = 1500
    audio_num_blocks: int = 32

    vocab_dim: int = 51866
    txt_seq_len: int = 448
    txt_input_dim: int = 1280 # must be the same as audio embedding size
    txt_num_heads: int = 4
    txt_num_blocks: int = 32


if __name__ == '__main__':
    weights1 = torch.load('whisper_01.bin')
    weights2 = torch.load('whisper_02.bin')
    weights = weights1
    weights.update(weights2)
    weights = {k.split('model.')[1]: v for k, v in weights.items() if k not in ['proj_out.weight', 'model.encoder.embed_positions.weight']}
    
    whisper = Whisper(WhisperParams())
    whisper.load_state_dict(weights)
