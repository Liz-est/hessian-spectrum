"""
Minimal single-layer LLaMA-style decoder for toy experiments.

Design choices (match the param-counting formulas in the README):
  - RMSNorm (no bias)                       -> 2 norms/block + 1 final = 3d params
  - Rotary positional embedding (RoPE)      -> no learned position params
  - Multi-head causal self-attention        -> wq/wk/wv/wo, each d x d  = 4 d^2
  - SwiGLU FFN (gate / up / down, no bias)  -> 3 * d * d_ff
  - Embedding and LM head are NOT tied      -> 2 * V * d

Parameter counts (single layer):
  N_embed+head = 2 * V * d
  N_transformer = 4 * d^2 + 3 * d * d_ff + 3 * d
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ToyLlamaConfig:
    vocab_size: int = 1024
    n_embd: int = 160          # hidden size d
    n_head: int = 5            # attention heads h
    head_dim: int = 32         # d_head (n_head * head_dim need not equal n_embd)
    n_ffn: int = 448           # FFN inner size d_ff
    n_layer: int = 1           # single-layer decoder
    block_size: int = 512      # max sequence length
    rope_theta: float = 10000.0
    dropout: float = 0.0
    device: str = "cpu"


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


def precompute_rope(head_dim, seqlen, theta, device):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seqlen, device=device).float()
    freqs = torch.outer(t, freqs)                       # (seqlen, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (bsz, n_head, seqlen, head_dim)
    seqlen = x.shape[-2]
    cos = cos[:seqlen].view(1, 1, seqlen, -1)
    sin = sin[:seqlen].view(1, 1, seqlen, -1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos
    out = torch.stack((rot_x1, rot_x2), dim=-1)
    return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        inner = config.n_head * config.head_dim
        self.wq = nn.Linear(config.n_embd, inner, bias=False)
        self.wk = nn.Linear(config.n_embd, inner, bias=False)
        self.wv = nn.Linear(config.n_embd, inner, bias=False)
        self.wo = nn.Linear(inner, config.n_embd, bias=False)

    def forward(self, x, cos, sin):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        xk = self.wk(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        xq = apply_rope(xq, cos, sin)
        xk = apply_rope(xk, cos, sin)

        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    """LLaMA-style gated FFN: down(silu(gate(x)) * up(x))."""
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_ffn, bias=False)
        self.up = nn.Linear(config.n_embd, config.n_ffn, bias=False)
        self.down = nn.Linear(config.n_ffn, config.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class ToyLlama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # untied

        cos, sin = precompute_rope(config.head_dim, config.block_size,
                                   config.rope_theta, torch.device(config.device))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

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
        embed_head = self.tok_emb.weight.numel() + self.lm_head.weight.numel()
        total = self.num_params()
        return {
            "total": total,
            "embed+head": embed_head,
            "transformer": total - embed_head,
        }

    def forward(self, idx, targets=None):
        b, t = idx.shape
        assert t <= self.config.block_size
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.norm_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
