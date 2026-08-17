"""
逐 **unit** 精确 GN/Fisher 子块（QuadraticModel-rep 版「Hessian 路线一」）。

与 layers.py（整张量/整层 + Lanczos 近似）互补：这里对**更细的单元**精确构造
GN/Fisher 块再 `torch.linalg.eigvalsh` 拿全部特征值——不用 Lanczos、无 Ritz 近似、
无误差带。数学同 toy_models/hessian_toy.py 的 NeuronHessian，但本模型参数是**跨层
stack 的叶子张量**、forward 用函数式 einsum（没有 nn.Linear 子模块可挂 hook），故
在这里**内联重算 forward**、对目标层的 einsum 输出 `retain_grad()`，再 `loss.backward()`
读 `.grad`。只需一阶反传（非 HVP），无 create_graph、无需 SDPA MATH 后端。

四类 unit（对 weight W: 输入 a_t → 输出 y_t=einsum(a_t,W)）：
  · 按输出神经元 i：H_i = (1/N) Σ_t (∂L/∂y_{i,t})² a_t a_tᵀ   （块 = d_in×d_in，PSD）
      → attn_v / attn_head(O) / mlp_up(fc) / mlp_head(proj)
  · 按注意头 h（Q/K）：H_h = (1/N) Σ_t a_t a_tᵀ ⊗ g_{h,t} g_{h,t}ᵀ
      = UᵀU/N，u_t = a_t ⊗ g_{h,t}                            （块 = (d_in·K)×(d_in·K)）
  · 按 token（embedding）：H_v = (1/N) Σ_{t:x_t=v} g_t g_tᵀ，g_t=∂L/∂emb_out_t
      用**全局** N 归一化 → 块幅度保留 N_v/N 词频缩放                   （块 = D×D）
  · 按 token=class（lm_head，CE 闭式）：G_{vv} = (1/T) Σ_t q_{t,v}(1−q_{t,v}) x_t x_tᵀ
      x_t=最后一层 rmsnorm 输出，q=softmax(logits)             （块 = D×D；复用 dense_block_eig）

各张量的「输入 a_t / 输出神经元轴」对照（见 model._block）：
    attn_q/k   a=attn_in(D)   输出 q/k_raw(H,K)   按 head 切（H=16 头）
    attn_v     a=attn_in(D)   输出 v_raw(H·K)     按输出神经元切（H·K=1024）
    attn_head  a=attn_agg(H·K) 输出 attn_out(D)   按输出神经元切（D=1024）
    mlp_up     a=mlp_in(D)    输出 u_raw(M)       按输出神经元切（M=4096）
    mlp_head   a=gelu_u(M)    输出 mlp_out(D)     按输出神经元切（D=1024）
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from model import rmsnorm, apply_rope


# 每个 unit 张量的「输出神经元数 / 头数」与块维（供编排/画图预估规模）
def unit_layout(cfg):
    D, H, K, M = cfg.D, cfg.H, cfg.K, cfg.M
    return {
        "attn_q":    {"kind": "head",   "n_units": H,      "d_block": D * K},
        "attn_k":    {"kind": "head",   "n_units": H,      "d_block": D * K},
        "attn_v":    {"kind": "neuron", "n_units": H * K,  "d_block": D},
        "attn_head": {"kind": "neuron", "n_units": D,      "d_block": H * K},
        "mlp_up":    {"kind": "neuron", "n_units": M,      "d_block": D},
        "mlp_head":  {"kind": "neuron", "n_units": D,      "d_block": M},
    }


class UnitHessian:
    """对给定 checkpoint 的模型，逐 unit 精确构造 GN/Fisher 块并特征分解。

    get_batch(): 无参可调用，每次返回一个 (X, Y) minibatch（int64，在 device 上）。
    n_batches:   过多少个 minibatch 累加（决定统计的 token 数 N）。
    device:      构块与 eigvalsh 的设备。
    cache_device: 缓存激活/梯度的设备（大 N 时设 'cpu' 省显存；默认同 device）。
    """
    def __init__(self, model, get_batch, n_batches=20, device="cpu", cache_device=None):
        self.model = model
        self.cfg = model.cfg
        self.get_batch = get_batch
        self.n_batches = n_batches
        self.device = torch.device(device)
        self.cache_device = torch.device(cache_device) if cache_device else self.device

    # ------------------------------------------------------------------
    # 内联 forward：重算一遍 model.forward，但把目标层 l 的 einsum 输出 retain_grad，
    # 以便 backward 后读 .grad。cap 里存的是**需要 grad 的中间张量**（含 retain_grad）。
    # want ⊆ {"emb","attn_qkv","attn_o","mlp_up","mlp_head","head"} 控制捕获哪些，
    # 未捕获的照常前向、不 retain_grad（省内存）。
    # ------------------------------------------------------------------
    def _forward(self, X, Y, l, want):
        cfg = self.cfg
        cap = {}
        h = self.model.embd[X]                                   # (B,T,D)
        if "emb" in want:
            h.retain_grad(); cap["emb_out"] = h

        for li in range(cfg.L):
            tgt = (li == l)
            # ---- attention ----
            hn = rmsnorm(h, cfg.norm_eps)
            if tgt and "attn_qkv" in want:
                cap["attn_in"] = hn.detach()                     # 输入激活（不需 grad）
            q = torch.einsum("...D,DHK->...HK", hn, self.model.attn_q[li])
            k = torch.einsum("...D,DHK->...HK", hn, self.model.attn_k[li])
            v = torch.einsum("...D,DHK->...HK", hn, self.model.attn_v[li])
            if tgt and "attn_qkv" in want:
                for t in (q, k, v):
                    t.retain_grad()
                cap["q_raw"], cap["k_raw"], cap["v_raw"] = q, k, v
            qn = apply_rope(rmsnorm(q, cfg.norm_eps), cfg.rope_freq)
            kn = apply_rope(rmsnorm(k, cfg.norm_eps), cfg.rope_freq)
            qt, kt, vt = (t.transpose(1, 2) for t in (qn, kn, v))
            a = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
            a = a.transpose(1, 2)                                # (B,T,H,K)
            if tgt and "attn_o" in want:
                cap["attn_agg"] = a.detach()                     # attn_head 的输入
            attn_out = torch.einsum("...HK,HKD->...D", a, self.model.attn_head[li])
            if tgt and "attn_o" in want:
                attn_out.retain_grad(); cap["attn_out"] = attn_out
            h = h + attn_out / cfg.L
            # ---- mlp ----
            hn = rmsnorm(h, cfg.norm_eps)
            if tgt and "mlp_up" in want:
                cap["mlp_in"] = hn.detach()
            u = torch.einsum("...D,DM->...M", hn, self.model.mlp_up[li])
            if tgt and "mlp_up" in want:
                u.retain_grad(); cap["u_raw"] = u
            gu = F.gelu(u, approximate="tanh")
            if tgt and "mlp_head" in want:
                cap["gelu_u"] = gu.detach()
            mlp_out = torch.einsum("...M,MD->...D", gu, self.model.mlp_head[li])
            if tgt and "mlp_head" in want:
                mlp_out.retain_grad(); cap["mlp_out"] = mlp_out
            h = h + mlp_out / cfg.L

        hn = rmsnorm(h, cfg.norm_eps)
        if "head" in want:
            cap["head_in"] = hn.detach()
        logits = torch.einsum("...TD,DV->...TV", hn, self.model.head)
        if "head" in want:
            cap["logits"] = logits.detach()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               Y.reshape(-1), ignore_index=-1)
        return loss, cap

    # ------------------------------------------------------------------
    # 按输出神经元：attn_v / attn_head / mlp_up / mlp_head
    #   块 i = (1/N) Σ_t (∂L/∂y_{i,t})² a_t a_tᵀ
    # 增量累加 Hsum (n_units, d_in, d_in)；d_in 小（≤4096），逐 batch 前向反传。
    # ------------------------------------------------------------------
    @torch.enable_grad()
    def neuron_blocks(self, param_name, l):
        cfg = self.cfg
        # (want, 输出张量键, 输入激活键, n_units=输出神经元数, d_in=输入维)
        spec = {
            "attn_v":    ("attn_qkv", "v_raw",    "attn_in",  cfg.H * cfg.K, cfg.D),
            "attn_head": ("attn_o",   "attn_out", "attn_agg", cfg.D,         cfg.H * cfg.K),
            "mlp_up":    ("mlp_up",   "u_raw",    "mlp_in",   cfg.M,         cfg.D),
            "mlp_head":  ("mlp_head", "mlp_out",  "gelu_u",   cfg.D,         cfg.M),
        }[param_name]
        want, out_key, in_key, n_units, d_in = spec

        Hsum = torch.zeros((n_units, d_in, d_in), dtype=torch.float64, device=self.device)
        n_tok = 0
        for _ in range(self.n_batches):
            X, Y = self.get_batch()
            self.model.zero_grad(set_to_none=True)
            loss, cap = self._forward(X, Y, l, {want})
            loss.backward()
            a = cap[in_key].reshape(-1, d_in).to(torch.float64).to(self.device)   # (N,d_in)
            g = cap[out_key].grad.reshape(-1, n_units).to(torch.float64).to(self.device)  # (N,n_units)
            N = a.shape[0]
            # 分块累加避免一次性 (n_units,N) 大张量；n_units 可达 4096
            for i in range(n_units):
                s = g[:, i] ** 2                                  # (N,)
                Hsum[i] += a.t() @ (s.unsqueeze(1) * a)
            n_tok += N
        self.model.zero_grad(set_to_none=True)
        Hsum /= max(1, n_tok)
        eigs = np.stack([torch.linalg.eigvalsh(Hsum[i]).cpu().numpy()
                         for i in range(n_units)])
        return eigs, {"unit": "neuron", "n_units": n_units, "d_block": d_in, "n_tok": n_tok}

    # ------------------------------------------------------------------
    # 按注意头：attn_q / attn_k
    #   块 h = (1/N) Σ_t u_t u_tᵀ，u_t = a_t ⊗ g_{h,t}  ∈ R^{D·K}
    # 缓存小激活 (a:(N,D), g:(N,H,K)) 跨 batch，再**逐头**建 (D·K)² 大块（fp32，17GB）。
    # ------------------------------------------------------------------
    @torch.enable_grad()
    def head_blocks(self, param_name, l):
        cfg = self.cfg
        D, H, K = cfg.D, cfg.H, cfg.K
        out_key = {"attn_q": "q_raw", "attn_k": "k_raw"}[param_name]
        dim = D * K

        a_list, g_list = [], []
        for _ in range(self.n_batches):
            X, Y = self.get_batch()
            self.model.zero_grad(set_to_none=True)
            loss, cap = self._forward(X, Y, l, {"attn_qkv"})
            loss.backward()
            a_list.append(cap["attn_in"].reshape(-1, D).to(torch.float32).to(self.cache_device))
            g_list.append(cap[out_key].grad.reshape(-1, H, K).to(torch.float32).to(self.cache_device))
        self.model.zero_grad(set_to_none=True)
        a = torch.cat(a_list, 0); g = torch.cat(g_list, 0)        # (N,D),(N,H,K)
        del a_list, g_list
        n_tok = a.shape[0]

        eigs = []
        for hh in range(H):
            # 逐头把 (N,D·K) 的 U 分批搬上 device 累加 UᵀU（避免一次性 N×dim）
            Hh = torch.zeros((dim, dim), dtype=torch.float32, device=self.device)
            gh = g[:, hh, :]                                      # (N,K)
            step = max(1, 200_000 // dim)                        # 每次 ~step 行
            for s in range(0, n_tok, step):
                ab = a[s:s + step].to(self.device)
                gb = gh[s:s + step].to(self.device)
                U = torch.einsum("ni,nj->nij", ab, gb).reshape(-1, dim)   # (b,D·K)
                Hh += U.t() @ U
                del U
            Hh /= max(1, n_tok)
            eigs.append(torch.linalg.eigvalsh(Hh.double()).cpu().numpy())
            del Hh
        return np.stack(eigs), {"unit": "head", "n_units": H, "d_block": dim, "n_tok": n_tok}

    # ------------------------------------------------------------------
    # embedding 按 token：H_v = (1/N_global) Σ_{t:x_t=v} g_t g_tᵀ，g_t=∂L/∂emb_out_t
    # ------------------------------------------------------------------
    @torch.enable_grad()
    def embedding_token_blocks(self, token_ids):
        cfg = self.cfg
        D = cfg.D
        sel = torch.as_tensor(np.asarray(token_ids), dtype=torch.long, device=self.device)
        n_sel = sel.numel()
        lut = torch.full((cfg.V,), -1, dtype=torch.long, device=self.device)
        lut[sel] = torch.arange(n_sel, device=self.device)
        Hsum = torch.zeros((n_sel, D, D), dtype=torch.float64, device=self.device)
        cnt = torch.zeros(n_sel, dtype=torch.float64, device=self.device)
        n_tok = 0                                                 # 全局位置计数
        for _ in range(self.n_batches):
            X, Y = self.get_batch()
            self.model.zero_grad(set_to_none=True)
            loss, cap = self._forward(X, Y, l=-1, want={"emb"})
            loss.backward()
            ids = X.reshape(-1).to(self.device)                   # (N,)
            g = cap["emb_out"].grad.reshape(-1, D).to(torch.float64).to(self.device)
            n_tok += ids.numel()
            slot = lut[ids]; mask = slot >= 0
            slot = slot[mask]; gm = g[mask]
            for u in torch.unique(slot):
                gv = gm[slot == u]                                # (N_v,D)
                Hsum[u] += gv.t() @ gv
            cnt.index_add_(0, slot, torch.ones_like(slot, dtype=torch.float64))
        self.model.zero_grad(set_to_none=True)
        Hsum /= max(1, n_tok)
        eigs = np.stack([torch.linalg.eigvalsh(Hsum[v]).cpu().numpy() for v in range(n_sel)])
        return eigs, {"unit": "token", "n_units": n_sel, "d_block": D,
                      "n_tok": n_tok, "counts": cnt.cpu().numpy()}

    # ------------------------------------------------------------------
    # lm_head 按 token(=class)，CE 闭式：复用 dense_block_eig.head_col_spectra。
    #   G_{vv} = (1/T) Σ_t q_{t,v}(1−q_{t,v}) x_t x_tᵀ    x_t=最后 rmsnorm 输出
    # ------------------------------------------------------------------
    @torch.no_grad()
    def lm_head_token_blocks(self, token_ids, chunk=8192):
        from dense_block_eig import head_col_spectra
        cfg = self.cfg
        Xs, Qs = [], []
        for _ in range(self.n_batches):
            X, Y = self.get_batch()
            _loss, cap = self._forward(X, Y, l=-1, want={"head"})
            Xs.append(cap["head_in"].reshape(-1, cfg.D).float().to(self.cache_device))
            Qs.append(F.softmax(cap["logits"].float(), -1).reshape(-1, cfg.V).to(self.cache_device))
        Xc = torch.cat(Xs); Qc = torch.cat(Qs)
        cols = [int(v) for v in np.asarray(token_ids)]
        spec = head_col_spectra(Xc, Qc, cols, self.device, chunk=chunk)   # {v: eigs}
        eigs = np.stack([spec[v] for v in cols])
        return eigs, {"unit": "token(class)", "n_units": len(cols),
                      "d_block": cfg.D, "n_tok": Xc.shape[0]}
