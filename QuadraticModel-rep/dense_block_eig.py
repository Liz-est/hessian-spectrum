"""
小块的**精确**谱：直接建稠密 H_bb 后 eigh，以及 lm_head 的闭式块。
不进 Lanczos 管线 —— 没有 Ritz 近似、没有误差带、不需要 gauss_radau。

何时用稠密而非 Lanczos：Lanczos 花 m 次 matvec 给你 m 个 Ritz 值（近似）；
用 n_b 次 matvec 可以精确建出整个 H_bb 拿到全部 n_b 个特征值。分界约在 n_b ≲ 2~3m。

lm_head 闭式（不需要任何自动微分）
--------------------------------
W:(D,V)，x_t = rmsnorm(h_t) ∈ R^D，q_t = softmax(logits_t) ∈ R^V，A_t = diag(q_t) − q_tq_tᵀ。
按 vec(W) 索引序 (d,v)（d 在外）：
    G = (1/T) Σ_t (x_t x_tᵀ) ⊗ A_t
这是 Kronecker **和**，一般不可分解。但两种切法各自塌掉：

  按 v 切（固定 v₀，参数 W[:,v₀] ∈ R^D，"每个词一块"）:
      G_{v₀v₀} = (1/T) Σ_t q_{t,v₀}(1 − q_{t,v₀}) · x_t x_tᵀ           (D×D)
    加权 Gram 矩阵，一次前向累加即得，D=1024 直接 eigh。

  按 d 切（固定 d₀，参数 W[d₀,:] ∈ R^V，"每个隐藏坐标一块"）:
      G_{d₀d₀} = diag(s) − (1/T) Σ_t x²_{t,d₀} q_t q_tᵀ,  s_v = (1/T) Σ_t x²_{t,d₀} q_{t,v}
    对角 + 低秩；V=8192 的稠密 eigh 可做，或用低秩结构。

  ⚠ 只有 A_t 与 t 无关时（MSE: A_t=(2/C)I）Kronecker 才精确，那时各 d 行的块互为
  标量倍、谱形状完全相同；CE 下 A_t 随 t 变，各行是对 {A_t} 的**不同加权平均**——
  行块谱的差异量化的正是"哪个隐藏坐标在给哪些 token 的曲率加权"。

用法:
  python dense_block_eig.py --ckpt checkpoints_b64/ckpt_p100.pt --mode head-cols --topk 64
  python dense_block_eig.py --ckpt ... --mode dense --groups layer00.head00
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layers import build_layers                              # noqa: E402
from hvp_layers import SharedBatchOp                         # noqa: E402
from model import rmsnorm                                    # noqa: E402
from spectrum_ddp import load_checkpoint, make_local_batches  # noqa: E402


# --------------------------------------------------------------------------
# 通用：用 n_b 次 matvec 精确建稠密 H_bb
# --------------------------------------------------------------------------
def dense_block(model, batches, nb_global, block, kind, precond, device):
    """返回稠密 H_bb (n_b × n_b)。代价 n_b 次 matvec —— 只对小块用。"""
    n = block.numel
    M = torch.zeros(n, n, dtype=torch.float64)
    eye = torch.eye(n, dtype=torch.float32, device=device)
    for x, y in batches:
        op = SharedBatchOp(model, x, y, kind, precond)
        op.prepare([block])
        for i in range(n):
            M[:, i] += op.apply(block, eye[i]).double().cpu()
        op.release()
    M /= nb_global
    return 0.5 * (M + M.T)      # 对称化，消掉数值不对称


# --------------------------------------------------------------------------
# lm_head 闭式：一次前向收集 (x_t, q_t)
# --------------------------------------------------------------------------
@torch.no_grad()
def collect_xq(model, cfg, batches, device):
    """收集 x_t = rmsnorm(h_t) 与 q_t = softmax(logits_t)。返回 (X:(T,D), Q:(T,V))。"""
    Xs, Qs = [], []
    for x, _y in batches:
        h = model.embd[x]
        for l in range(cfg.L):
            h = model._block(h, l)
        hn = rmsnorm(h, cfg.norm_eps)
        logits = torch.einsum("...TD,DV->...TV", hn, model.head)
        Xs.append(hn.reshape(-1, cfg.D).float().cpu())
        Qs.append(F.softmax(logits.float(), dim=-1).reshape(-1, cfg.V).cpu())
    return torch.cat(Xs), torch.cat(Qs)


def head_col_spectra(X, Q, cols, device, chunk=8192):
    """按 v 切：G_{vv} = (1/T) Σ_t q_{t,v}(1−q_{t,v}) x_t x_tᵀ  (D×D)，精确 eigh。"""
    T, D = X.shape
    out = {}
    for v in cols:
        G = torch.zeros(D, D, dtype=torch.float64, device=device)
        for s in range(0, T, chunk):
            Xc = X[s:s + chunk].to(device).double()
            qc = Q[s:s + chunk, v].to(device).double()
            G += (Xc * (qc * (1 - qc)).unsqueeze(1)).T @ Xc
        G /= T
        out[int(v)] = torch.linalg.eigvalsh(0.5 * (G + G.T)).cpu().numpy()
    return out


def head_row_spectra(X, Q, rows, device, chunk=8192):
    """按 d 切：G_{dd} = diag(s) − (1/T) Σ_t x²_{t,d} q_t q_tᵀ  (V×V)，精确 eigh。"""
    T, V = Q.shape
    out = {}
    for d in rows:
        s_vec = torch.zeros(V, dtype=torch.float64, device=device)
        low = torch.zeros(V, V, dtype=torch.float64, device=device)
        for s in range(0, T, chunk):
            qc = Q[s:s + chunk].to(device).double()
            w = (X[s:s + chunk, d].to(device).double()) ** 2
            s_vec += (qc * w.unsqueeze(1)).sum(0)
            low += (qc * w.unsqueeze(1)).T @ qc
        G = torch.diag(s_vec / T) - low / T
        out[int(d)] = torch.linalg.eigvalsh(0.5 * (G + G.T)).cpu().numpy()
    return out


def summarize(ev):
    ev = np.asarray(ev)
    pos = ev[ev > 0]
    return {"n": int(ev.size), "lam_max": float(ev.max()), "lam_min": float(ev.min()),
            "trace": float(ev.sum()),
            "eff_rank": float(pos.sum() ** 2 / (pos ** 2).sum()) if pos.size else 0.0,
            "n_pos": int((ev > 0).sum()), "n_neg": int((ev < 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", default="head-cols",
                    choices=["head-cols", "head-rows", "dense"])
    ap.add_argument("--groups", default="layer00.head00", help="--mode dense 用")
    ap.add_argument("--kind", default="gn", choices=["gn", "hessian"])
    ap.add_argument("--precond", default="raw", choices=["adam", "raw", "none"])
    ap.add_argument("--topk", type=int, default=64,
                    help="head-cols/rows：按频次取前 k 个（0=全部，V=8192 会很慢）")
    ap.add_argument("--n_tokens", type=int, default=262_144)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--ema", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model, cfg, precond, precond_raw = load_checkpoint(args.ckpt, args.ema, device)
    p = {"adam": precond, "raw": precond_raw, "none": None}[args.precond]

    batches, nb_global = make_local_batches(
        cfg, args.n_tokens, 1, 0, device, args.seed, per=args.per)
    print(f"batches={len(batches)} nb_global={nb_global} "
          f"tokens={nb_global*args.per*cfg.seq_len:,}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    npz, summary = {}, {}

    if args.mode in ("head-cols", "head-rows"):
        if args.precond != "none":
            print("⚠ 闭式路径当前只实现无预条件的 GN 块；--precond 被忽略", flush=True)
        X, Q = collect_xq(model, cfg, batches, device)
        print(f"收集 {X.shape[0]:,} token  X{tuple(X.shape)} Q{tuple(Q.shape)}", flush=True)
        if args.mode == "head-cols":
            # 按经验词频排序取 top-k（最不平衡的那些词最有意思）
            freq = Q.sum(0)
            idx = torch.argsort(freq, descending=True)
            cols = idx[:args.topk].tolist() if args.topk > 0 else list(range(cfg.V))
            spec = head_col_spectra(X, Q, cols, device)
            for v, ev in spec.items():
                npz[f"col{v:05d}_eigs"] = ev
                summary[f"col{v}"] = summarize(ev)
        else:
            rows = list(range(min(args.topk, cfg.D))) if args.topk > 0 else list(range(cfg.D))
            spec = head_row_spectra(X, Q, rows, device)
            for d, ev in spec.items():
                npz[f"row{d:05d}_eigs"] = ev
                summary[f"row{d}"] = summarize(ev)
    else:
        for b in build_layers(model, args.groups):
            print(f"稠密建块 {b.name} n_b={b.numel:,} "
                  f"（{b.numel} 次 matvec）", flush=True)
            M = dense_block(model, batches, nb_global, b, args.kind, p, device)
            ev = np.linalg.eigvalsh(M.numpy())
            npz[f"{b.name}_eigs"] = ev
            npz[f"{b.name}_dense"] = M.numpy().astype(np.float32)
            summary[b.name] = summarize(ev)

    np.savez(args.out, **npz)
    with open(args.out.replace(".npz", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for k, s in list(summary.items())[:16]:
        print(f"  {k:12s} λmax={s['lam_max']:.4e}  tr={s['trace']:.4e}  "
              f"eff_rank={s['eff_rank']:.1f}  neg={s['n_neg']}", flush=True)
    print(f"→ {args.out}", flush=True)


if __name__ == "__main__":
    main()
