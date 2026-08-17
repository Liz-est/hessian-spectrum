"""
逐 unit 精确块（blocks.UnitHessian）的正确性核对：闭式 vs **独立的扰动-叶子 autograd**。

blocks.py 的块是「对位置对角」的经验 Fisher / GN：对 weight W（a_t → y_t=W·a_t），
  · 按输出神经元 i：H_i = (1/N) Σ_t g_{i,t}² a_t a_tᵀ,     g_{i,t}=∂L/∂y_{i,t}
  · 按头 h（Q/K）：  H_h = (1/N) Σ_t (a_t⊗g_{h,t})(·)ᵀ
  · 按 token（emb）： H_v = (1/N) Σ_{t:x_t=v} g_t g_tᵀ,      g_t=∂L/∂emb_out_t
  · lm_head 按 class：G_vv = (1/T) Σ_t p_{t,v}(1−p_{t,v}) x_t x_tᵀ  （CE 真 Hessian）

⚠ 关键：这是「对位置 t **对角**」的量（丢掉 s≠t 交叉项），**不等于**逐 token 对参数的
梯度外积 ∂ℓ_t/∂W（后者对 W 影响到的所有位置求和）。故独立 oracle 必须直接拿
g_{i,t}=∂L/∂y_{i,t}：在目标 einsum 输出处注入**零值叶子 E**（y_used=y+E），backward 后
E.grad 即逐位置逐神经元的 ∂L/∂y——与 blocks.py 的 retain_grad 路径完全独立。a_t 亦独立重算。

跑：python verify_blocks.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from model import Transformer, TransformerConfig, rmsnorm, apply_rope
from blocks import UnitHessian


def eig_close(name, e_closed, e_brute, atol=1e-6, rtol=1e-5):
    e1 = np.sort(np.asarray(e_closed).ravel())
    e2 = np.sort(np.asarray(e_brute).ravel())
    d = np.abs(e1 - e2).max()
    scale = max(1.0, np.abs(e2).max())
    ok = d <= atol + rtol * scale
    print(f"  [{'OK ' if ok else 'FAIL'}] {name:30s} max|Δλ|={d:.2e}  "
          f"λmax={e2.max():.3e}  n={e1.size}")
    return ok


def fwd_with_leaves(model, X, Y):
    """独立重算 forward，在每个感兴趣的 einsum 输出处注入零叶子 E_*（y_used=y+E）。
    返回 (loss, cap)：cap 存输入激活 a_*（detach）与叶子 E_*（backward 后读 .grad）。
    只做 l=0 与 l=1 两层的目标输出 + emb + head，够本测试用。"""
    cfg = model.cfg
    cap = {}

    def leaf(y, key):
        E = torch.zeros_like(y, requires_grad=True)
        cap[key] = E
        return y + E

    h = model.embd[X]                                        # (B,T,D)
    cap["emb_out"] = leaf_h = torch.zeros_like(h, requires_grad=True)
    h = h + leaf_h                                           # emb_out 叶子
    for li in range(cfg.L):
        hn = rmsnorm(h, cfg.norm_eps)
        cap[f"attn_in{li}"] = hn.detach()
        q = torch.einsum("...D,DHK->...HK", hn, model.attn_q[li])
        k = torch.einsum("...D,DHK->...HK", hn, model.attn_k[li])
        v = torch.einsum("...D,DHK->...HK", hn, model.attn_v[li])
        q = leaf(q, f"q_raw{li}"); k = leaf(k, f"k_raw{li}"); v = leaf(v, f"v_raw{li}")
        qn = apply_rope(rmsnorm(q, cfg.norm_eps), cfg.rope_freq)
        kn = apply_rope(rmsnorm(k, cfg.norm_eps), cfg.rope_freq)
        qt, kt, vt = (t.transpose(1, 2) for t in (qn, kn, v))
        with sdpa_kernel([SDPBackend.MATH]):
            a = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
        a = a.transpose(1, 2)
        cap[f"attn_agg{li}"] = a.detach()
        attn_out = torch.einsum("...HK,HKD->...D", a, model.attn_head[li])
        attn_out = leaf(attn_out, f"attn_out{li}")
        h = h + attn_out / cfg.L
        hn = rmsnorm(h, cfg.norm_eps)
        cap[f"mlp_in{li}"] = hn.detach()
        u = torch.einsum("...D,DM->...M", hn, model.mlp_up[li])
        u = leaf(u, f"u_raw{li}")
        gu = F.gelu(u, approximate="tanh")
        cap[f"gelu_u{li}"] = gu.detach()
        mlp_out = torch.einsum("...M,MD->...D", gu, model.mlp_head[li])
        mlp_out = leaf(mlp_out, f"mlp_out{li}")
        h = h + mlp_out / cfg.L
    hn = rmsnorm(h, cfg.norm_eps)
    logits = torch.einsum("...TD,DV->...TV", hn, model.head)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Y.reshape(-1),
                           ignore_index=-1)
    return loss, cap


def brute_neuron(cap, in_key, out_key, n_units, d_in, N, sel):
    a = cap[in_key].reshape(-1, d_in).double()               # (N,d_in)
    g = cap[out_key].grad.reshape(-1, n_units).double()      # (N,n_units)
    out = []
    for i in sel:
        s = g[:, i] ** 2
        H = (a.t() @ (s.unsqueeze(1) * a)) / N
        out.append(torch.linalg.eigvalsh(H).numpy())
    return np.stack(out)


def brute_head(cap, in_key, out_key, D, H, K, N, heads):
    a = cap[in_key].reshape(-1, D).double()                  # (N,D)
    g = cap[out_key].grad.reshape(-1, H, K).double()         # (N,H,K)
    dim = D * K
    out = []
    for h in heads:
        U = torch.einsum("ni,nj->nij", a, g[:, h, :]).reshape(N, dim)
        M = (U.t() @ U) / N
        out.append(torch.linalg.eigvalsh(M).numpy())
    return np.stack(out)


def brute_embedding(cap, X, D, N, token_ids):
    g = cap["emb_out"].grad.reshape(-1, D).double()          # (N,D)
    ids = X.reshape(-1)
    out = []
    for v in token_ids:
        m = (ids == v)
        gv = g[m]
        H = (gv.t() @ gv) / N
        out.append(torch.linalg.eigvalsh(H).numpy())
    return np.stack(out)


def brute_lmhead_token(model, X, Y, token_ids):
    """CE 真 Hessian（对整块 head double-backward，再切出第 v 列的 D×D 子块）。
    head 形状 (D,V)，vec 索引 (d,v)（d 在外）→ 第 v 列参数是 head[:,v]，其自身块 = 每行
    ∂²L/∂head[d,v]∂head[·,v]。逐 d 对 head 求二阶梯度、取该列切片。"""
    D, V = model.cfg.D, model.cfg.V
    with sdpa_kernel([SDPBackend.MATH]):
        _, loss = model(X, Y)
    g = torch.autograd.grad(loss, model.head, create_graph=True)[0]   # (D,V)
    out = []
    for v in token_ids:
        rows = []
        for d in range(D):
            hd = torch.autograd.grad(g[d, v], model.head, retain_graph=True)[0]  # (D,V)
            rows.append(hd[:, v])                     # 只取同列 → D×D 块的一行
        Hm = torch.stack(rows).double()
        out.append(torch.linalg.eigvalsh(0.5 * (Hm + Hm.T)).numpy())
    return np.stack(out)


def main():
    torch.manual_seed(0)
    cfg = TransformerConfig(D=16, L=2, M=32, H=2, K=8, V=24, seq_len=6)
    m = Transformer(cfg)
    with torch.no_grad():
        m.head.normal_(0, 0.15)
    m.eval()

    X = torch.randint(0, cfg.V, (2, cfg.seq_len))
    Y = torch.randint(0, cfg.V, (2, cfg.seq_len))
    N = X.numel()
    uh = UnitHessian(m, lambda: (X, Y), n_batches=1, device="cpu")

    # 独立扰动-叶子前向（一次 backward 拿全部 ∂L/∂y）
    loss, cap = fwd_with_leaves(m, X, Y)
    loss.backward()

    res = []

    # 1) neuron mlp_up (l0)：out=u_raw0(M), in=mlp_in0(D)
    e_c, _ = uh.neuron_blocks("mlp_up", l=0)
    sel = [0, 3, 7]
    e_b = brute_neuron(cap, "mlp_in0", "u_raw0", cfg.M, cfg.D, N, sel)
    res.append(eig_close("neuron mlp_up (l0)", e_c[sel], e_b))

    # 2) neuron attn_head O (l1)：out=attn_out1(D), in=attn_agg1(H·K)
    e_c, _ = uh.neuron_blocks("attn_head", l=1)
    sel2 = [0, 5, 11]
    e_b = brute_neuron(cap, "attn_agg1", "attn_out1", cfg.D, cfg.H * cfg.K, N, sel2)
    res.append(eig_close("neuron attn_head O (l1)", e_c[sel2], e_b))

    # 3) neuron attn_v (l0)：out=v_raw0(H·K), in=attn_in0(D)
    e_c, _ = uh.neuron_blocks("attn_v", l=0)
    sel3 = [0, 4, 9]
    e_b = brute_neuron(cap, "attn_in0", "v_raw0", cfg.H * cfg.K, cfg.D, N, sel3)
    res.append(eig_close("neuron attn_v (l0)", e_c[sel3], e_b))

    # 4) neuron mlp_head (l1)：out=mlp_out1(D), in=gelu_u1(M)
    e_c, _ = uh.neuron_blocks("mlp_head", l=1)
    sel4 = [0, 5, 11]
    e_b = brute_neuron(cap, "gelu_u1", "mlp_out1", cfg.D, cfg.M, N, sel4)
    res.append(eig_close("neuron mlp_head (l1)", e_c[sel4], e_b))

    # 5) head attn_q (l0)：out=q_raw0(H,K), in=attn_in0(D)
    e_c, _ = uh.head_blocks("attn_q", l=0)
    e_b = brute_head(cap, "attn_in0", "q_raw0", cfg.D, cfg.H, cfg.K, N, [0, 1])
    res.append(eig_close("head attn_q (l0)", e_c, e_b))

    # 6) head attn_k (l1)：out=k_raw1(H,K), in=attn_in1(D)
    e_c, _ = uh.head_blocks("attn_k", l=1)
    e_b = brute_head(cap, "attn_in1", "k_raw1", cfg.D, cfg.H, cfg.K, N, [0, 1])
    res.append(eig_close("head attn_k (l1)", e_c, e_b))

    # 7) embedding token
    toks = [int(t) for t in torch.unique(X.reshape(-1))[:4]]
    e_c, _ = uh.embedding_token_blocks(token_ids=toks)
    e_b = brute_embedding(cap, X, cfg.D, N, toks)
    res.append(eig_close(f"embedding token", e_c, e_b))

    # 8) lm_head token（CE 真 Hessian double-backward）
    toks2 = [0, 1, 2]
    e_c, _ = uh.lm_head_token_blocks(token_ids=toks2)
    e_b = brute_lmhead_token(m, X, Y, toks2)
    res.append(eig_close(f"lm_head token", e_c, e_b))

    print("\n" + ("✅ 全部通过" if all(res) else "❌ 有失败项"))
    return 0 if all(res) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
