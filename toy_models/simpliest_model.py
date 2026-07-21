"""
Zero-layer vanilla decoder-only Transformer for toy experiments.

Design choices (match the param-counting formulas in the README):
  - Post-LayerNorm (weight + bias)          -> 2 LN/block, each 2d = 4d params
  - Fixed sinusoidal position encoding       -> no trainable position params
  - Multi-head causal self-attention (bias)  -> 4 d^2 + 4d
  - FFN: Linear-ReLU-Linear (bias)           -> 2 d*d_ff + d_ff + d
  - LM Head with bias, NOT tied to embedding -> V*d + V
  - Token embedding                          -> V*d

Parameter counts (single layer, bias on):
  N_embed+head  = 2*V*d + V
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ToyVanillaConfig:
    vocab_size: int = 1024
    n_embd: int = 192          # hidden size d
    n_head: int = 6            # attention heads h
    head_dim: int = 32         # d_head (h * head_dim == n_embd here)
    n_ffn: int = 1024          # FFN inner size d_ff
    n_layer: int = 0           # single-layer decoder
    block_size: int = 128      # context length
    dropout: float = 0.0
    attn_dropout: float = 0.0
    device: str = "cpu"


def sinusoidal_encoding(seqlen, dim, device):
    """Fixed (non-trainable) sinusoidal position encoding, shape (seqlen, dim)."""
    pos = torch.arange(seqlen, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device).float()
                    * (-math.log(10000.0) / dim))
    pe = torch.zeros(seqlen, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.attn_dropout = config.attn_dropout
        inner = config.n_head * config.head_dim
        self.wq = nn.Linear(config.n_embd, inner, bias=True)
        self.wk = nn.Linear(config.n_embd, inner, bias=True)
        self.wv = nn.Linear(config.n_embd, inner, bias=True)
        self.wo = nn.Linear(inner, config.n_embd, bias=True)

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        xk = self.wk(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            xq, xk, xv, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class FFN(nn.Module):
    """Vanilla Linear-ReLU-Linear FFN."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_ffn, bias=True)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(config.n_ffn, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.relu(self.c_fc(x))))


class Block(nn.Module):
    """Post-LayerNorm block: x = LN(x + sublayer(x))."""
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.mlp = FFN(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.mlp(x))
        return x


class ToyVanilla(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)  # untied

        pe = sinusoidal_encoding(config.block_size, config.n_embd,
                                 torch.device(config.device))
        self.register_buffer("pos_enc", pe, persistent=False)  # fixed, non-trainable

        self.apply(self._init_weights)
        print(f"number of parameters: {self.num_params()/1e6:.3f}M")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self):
        tok = self.tok_emb.weight.numel()
        head = sum(p.numel() for p in self.lm_head.parameters())
        attn, ffn, ln = 0, 0, 0
        for blk in self.blocks:
            attn += sum(p.numel() for p in blk.attn.parameters())
            ffn += sum(p.numel() for p in blk.mlp.parameters())
            ln += sum(p.numel() for p in blk.ln_1.parameters())
            ln += sum(p.numel() for p in blk.ln_2.parameters())
        total = self.num_params()
        return {
            "token_embedding": tok,
            "lm_head": head,
            "embed+head": tok + head,
            "self_attention": attn,
            "ffn": ffn,
            "layernorm": ln,
            "transformer": attn + ffn + ln,
            "total": total,
        }

    def forward(self, idx, targets=None):
        b, t = idx.shape
        assert t <= self.config.block_size
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
