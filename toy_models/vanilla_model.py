"""
Minimal single-layer vanilla decoder-only Transformer for toy experiments.

Design choices (match the param-counting formulas in the README):
  - Every block uses pre-RMSNorm             -> scale only, no affine bias
  - Final RMSNorm after the block stack       -> only when n_layer > 0
  - Fixed sinusoidal position encoding       -> no trainable position params
  - Multi-head causal self-attention          -> configurable Linear bias
  - FFN: Linear-ReLU-Linear                   -> configurable Linear bias
  - LM Head, NOT tied to embedding            -> configurable Linear bias
  - Token embedding                          -> V*d

Default parameter counts for a Transformer block:
  N_embed+head  = 2*V*d
  N_transformer = 4*d^2 + 2*d*d_ff + 2d
      where 2d is the pair of RMSNorm scale vectors.
  N_final_norm  = d when n_layer > 0
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
    n_layer: int = 1           # single-layer decoder
    block_type: str = "transformer"  # "transformer" (attn+FFN) | "mlp" (FFN+FFN)
    block_size: int = 128      # context length
    dropout: float = 0.0
    attn_dropout: float = 0.0
    linear_bias: bool = False  # one policy for attention/FFN/lm_head Linear modules
    norm_eps: float = 1e-6
    loss_type: str = "ce"      # "ce" (softmax cross-entropy) | "mse" (vs one-hot)
    use_pos_enc: bool = True   # False -> skip adding the sinusoidal pos_enc in forward
    tok_emb_init_mean: float = 0.0
    tok_emb_init_std: float = 0.02
    lm_head_init_mean: float = 0.0
    lm_head_init_std: float = 0.02
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


class RMSNorm(nn.Module):
    """RMS normalization with one learned scale vector and no bias."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.attn_dropout = config.attn_dropout
        inner = config.n_head * config.head_dim
        self.wq = nn.Linear(config.n_embd, inner, bias=config.linear_bias)
        self.wk = nn.Linear(config.n_embd, inner, bias=config.linear_bias)
        self.wv = nn.Linear(config.n_embd, inner, bias=config.linear_bias)
        self.wo = nn.Linear(inner, config.n_embd, bias=config.linear_bias)

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
        self.c_fc = nn.Linear(config.n_embd, config.n_ffn, bias=config.linear_bias)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(config.n_ffn, config.n_embd, bias=config.linear_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.relu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-normalized block with no normalization bias.

    The first sub-layer is attention for a "transformer" block, or a second FFN
    for an "mlp" block (an MLP-only model). Both have the same n_embd->n_embd
    shape, so it is a drop-in swap and forward() is identical either way. The
    attribute stays `self.attn` so checkpoints and the analyzer's blocks.<i>.attn
    path prefix are unchanged regardless of block_type. Both variants use
    RMSNorm, with no LayerNorm or normalization bias anywhere.
    """
    def __init__(self, config):
        super().__init__()
        self.attn = FFN(config) if config.block_type == "mlp" \
            else CausalSelfAttention(config)
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = FFN(config)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ToyVanilla(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # A final norm is standard in pre-norm Transformers. Keep n_layer=0 as
        # the exact embed->lm_head model used by the full-batch experiment.
        self.final_norm = (
            RMSNorm(config.n_embd, config.norm_eps)
            if config.n_layer > 0
            else nn.Identity()
        )
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=config.linear_bias
        )  # untied

        pe = sinusoidal_encoding(config.block_size, config.n_embd,
                                 torch.device(config.device))
        self.register_buffer("pos_enc", pe, persistent=False)  # fixed, non-trainable

        if config.tok_emb_init_std < 0 or config.lm_head_init_std < 0:
            raise ValueError("initialization std must be non-negative")
        self.apply(self._init_weights)
        print(
            "initialization: "
            f"tok_emb=Normal({config.tok_emb_init_mean}, {config.tok_emb_init_std}), "
            f"lm_head=Normal({config.lm_head_init_mean}, {config.lm_head_init_std})"
        )
        print(f"number of parameters: {self.num_params()/1e6:.3f}M")

    def _init_weights(self, module):
        if module is self.tok_emb:
            torch.nn.init.normal_(
                module.weight,
                mean=self.config.tok_emb_init_mean,
                std=self.config.tok_emb_init_std,
            )
        elif module is self.lm_head:
            torch.nn.init.normal_(
                module.weight,
                mean=self.config.lm_head_init_mean,
                std=self.config.lm_head_init_std,
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            # Transformer-block matrices use the standard zero-mean Normal.
            # The full-batch n_layer=0 model has no such modules.
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
        ln += sum(p.numel() for p in self.final_norm.parameters())
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
        x = self.tok_emb(idx)
        if self.config.use_pos_enc:
            x = x + self.pos_enc[:t]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        if targets is not None:
            logits = self.lm_head(x)
            if self.config.loss_type == "mse":
                flat = logits.view(-1, logits.size(-1))
                tgt = targets.view(-1)
                mask = tgt != -1                       # drop ignore_index=-1
                flat = flat[mask]
                tgt = tgt[mask]
                onehot = F.one_hot(tgt, num_classes=flat.size(-1)).to(flat.dtype)
                loss = F.mse_loss(flat, onehot)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
