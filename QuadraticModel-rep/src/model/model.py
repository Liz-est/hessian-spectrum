"""
QuadraticModel Transformer 的 PyTorch 移植（OLMo-style）。

严格对齐 JAX 参考 QuadraticModel/transformer.py + ops.py：
  - RMSNorm 无可学习增益，eps=1e-6，float32 内部计算
  - QK-norm：对 q,k 的 head 维 K 做 weightless RMSNorm，**在 RoPE 之前**
  - RoPE：split-half 约定（前一半实部、后一半虚部），base=freq^(-idx/half_K)
  - GELU：tanh 近似（JAX jax.nn.gelu 默认 approximate=True）
  - 1/L 残差缩放，pre-norm
  - head（unembed）零初始化；无任何 bias
  - attention scale 默认 1/sqrt(K)，causal

参数命名与 JAX 树保持一致以便 checkpoint 互转 / eigenvector-mass 分组：
  embd                         (V, D)
  blocks.{l}.attn.{q,k,v}      (D, H, K)
  blocks.{l}.attn.head         (H, K, D)
  blocks.{l}.mlp.up            (D, M)
  blocks.{l}.mlp.head          (M, D)
  head                         (D, V)

CompleteP LR 乘子（pre/post）见 lrs()；训练器按参数组应用。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


@dataclass
class TransformerConfig:
    D: int = 1024
    L: int = 12
    M: int = 4096
    H: int = 16
    K: int = 64
    V: int = 8192
    seq_len: int = 1024
    norm_eps: float = 1e-6
    rope_freq: int = 10_000
    embd_scale: float = 1 / 64      # BlockScales.embd
    blocks_scale: float = 1.0
    head_scale: float = 1.0


def rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """weightless RMSNorm，内部 float32，输出还原原 dtype。"""
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf.to(dt)


def apply_rope(x: torch.Tensor, freq: float) -> torch.Tensor:
    """
    x: (..., T, H, K)，split-half RoPE，与 JAX ops.apply_rope 一致。
    在 float32 上计算再还原。
    """
    dt = x.dtype
    xf = x.float()
    T, _, K = xf.shape[-3:]
    half = K // 2
    idx = torch.arange(half, device=x.device, dtype=torch.float32)
    base = freq ** (-idx / half)                       # (half,)
    pos = torch.arange(T, device=x.device, dtype=torch.float32)
    theta = pos[:, None] * base[None, :]               # (T, half)
    cos = torch.cos(theta)[:, None, :]                 # (T,1,half)
    sin = torch.sin(theta)[:, None, :]
    xr, xi = xf[..., :half], xf[..., half:]            # 复数实/虚部
    out_r = xr * cos - xi * sin
    out_i = xr * sin + xi * cos
    return torch.cat([out_r, out_i], dim=-1).to(dt)


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        D, L, M, H, K, V = cfg.D, cfg.L, cfg.M, cfg.H, cfg.K, cfg.V
        p = nn.Parameter
        self.embd = p(torch.empty(V, D))
        self.attn_q = p(torch.empty(L, D, H, K))
        self.attn_k = p(torch.empty(L, D, H, K))
        self.attn_v = p(torch.empty(L, D, H, K))
        self.attn_head = p(torch.empty(L, H, K, D))
        self.mlp_up = p(torch.empty(L, D, M))
        self.mlp_head = p(torch.empty(L, M, D))
        self.head = p(torch.empty(D, V))
        self.reset_parameters()

    def reset_parameters(self):
        cfg = self.cfg
        D, M, H, K = cfg.D, cfg.M, cfg.H, cfg.K
        with torch.no_grad():
            self.embd.normal_(0, 1.0)
            self.attn_q.normal_(0, 1 / sqrt(D))
            self.attn_k.normal_(0, 1 / sqrt(D))
            self.attn_v.normal_(0, 1 / sqrt(D))
            self.attn_head.normal_(0, sqrt(D) / (H * K))
            self.mlp_up.normal_(0, 1 / sqrt(D))
            self.mlp_head.normal_(0, sqrt(D) / M)
            self.head.zero_()                           # 零初始化 unembed

    # ---- CompleteP 学习率乘子（pre, post），含 BlockScales ----
    def lr_groups(self):
        cfg = self.cfg
        D, L, M, H, K = cfg.D, cfg.L, cfg.M, cfg.H, cfg.K
        es, bs, hs = cfg.embd_scale, cfg.blocks_scale, cfg.head_scale
        # (name, param, pre, post*scale)
        return [
            ("embd",      self.embd,      D,       1 * es),
            ("attn_q",    self.attn_q,    L*H*K,   (1/D) * bs),
            ("attn_k",    self.attn_k,    L*H*K,   (1/D) * bs),
            ("attn_v",    self.attn_v,    L*H*K,   (1/D) * bs),
            ("attn_head", self.attn_head, L*D,     (1/(K*H)) * bs),
            ("mlp_up",    self.mlp_up,    L*M,     (1/D) * bs),
            ("mlp_head",  self.mlp_head,  L*D,     (1/M) * bs),
            ("head",      self.head,      1,       (1/D) * hs),
        ]

    def _block(self, h, l):
        cfg = self.cfg
        # attention
        hn = rmsnorm(h, cfg.norm_eps)
        q = torch.einsum("...D,DHK->...HK", hn, self.attn_q[l])
        k = torch.einsum("...D,DHK->...HK", hn, self.attn_k[l])
        v = torch.einsum("...D,DHK->...HK", hn, self.attn_v[l])
        q = apply_rope(rmsnorm(q, cfg.norm_eps), cfg.rope_freq)   # QK-norm before RoPE
        k = apply_rope(rmsnorm(k, cfg.norm_eps), cfg.rope_freq)
        # sdpa expects (B, H, T, K)
        qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)  # scale=1/sqrt(K)
        a = a.transpose(1, 2)                                     # (B,T,H,K)
        attn_out = torch.einsum("...HK,HKD->...D", a, self.attn_head[l])
        h = h + attn_out / cfg.L
        # mlp
        hn = rmsnorm(h, cfg.norm_eps)
        u = torch.einsum("...D,DM->...M", hn, self.mlp_up[l])
        u = F.gelu(u, approximate="tanh")
        mlp_out = torch.einsum("...M,MD->...D", u, self.mlp_head[l])
        h = h + mlp_out / cfg.L
        return h

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        """
        x: (B, T) token ids。返回 (logits, loss)。
        logits 始终为完整序列 (B, T, V)（Hessian 分析需要）。
        """
        cfg = self.cfg
        h = self.embd[x]                                # (B,T,D) 嵌入查表
        for l in range(cfg.L):
            h = self._block(h, l)
        h = rmsnorm(h, cfg.norm_eps)
        logits = torch.einsum("...TD,DV->...TV", h, self.head)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# 参数组 -> 扁平索引 slice（用于 eigenvector-mass / 预条件器分组）
def param_group_slices(model: "Transformer"):
    slices, off = {}, 0
    for name, p in model.named_parameters():
        n = p.numel()
        slices[name] = slice(off, off + n)
        off += n
    return slices, off


if __name__ == "__main__":
    cfg = TransformerConfig()
    m = Transformer(cfg)
    npар = m.n_params()
    print(f"n_params = {npар:,}  (expected 167,772,160)")
    # 逐组计数
    for name, p in m.named_parameters():
        print(f"  {name:12s} {tuple(p.shape)}  {p.numel():,}")
    x = torch.randint(0, cfg.V, (2, cfg.seq_len))
    with sdpa_kernel([SDPBackend.MATH]):
        logits, loss = m(x, x)
    print("logits", tuple(logits.shape), "loss", float(loss))
