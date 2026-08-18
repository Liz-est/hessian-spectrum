"""
块 matvec 的正确性验证（小模型，与暴力 autograd Hessian 对拍）。

核对三件事：
  1. H_bb v_b  ==  (暴力全 Hessian)[blk][:,blk] @ v_b     ← 投影语义正确
  2. GN 块同理（GN 用 J A Jᵀ 的暴力构造）
  3. lockstep（K 块共享 prepare）与逐块单独 prepare 结果逐位一致
  4. (H²)_bb ≠ (H_bb)² —— 定量确认「块谱不可从全量谱反推」

跑：python verify_blocks.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.model.model import Transformer, TransformerConfig
from src.model.layers import build_layers, full_block, check_partition
from src.spectrum.hvp_layers import SharedBatchOp
from src.spectrum.lanczos_layers import block_seed


def brute_hessian(model, x, y):
    """暴力全 Hessian（按 named_parameters flat 顺序）。"""
    params = list(model.parameters())
    n = sum(p.numel() for p in params)
    with sdpa_kernel([SDPBackend.MATH]):
        _, loss = model(x, y)
    g = torch.autograd.grad(loss, params, create_graph=True)
    gflat = torch.cat([t.reshape(-1) for t in g])
    rows = []
    for i in range(n):
        r = torch.autograd.grad(gflat[i], params, retain_graph=True, allow_unused=True)
        rows.append(torch.cat([
            (torch.zeros_like(p).reshape(-1) if t is None else t.reshape(-1))
            for t, p in zip(r, params)]))
    return torch.stack(rows)


def brute_gn(model, x, y):
    """暴力 GN = (1/T) Σ_t J_tᵀ A_t J_t，A_t = diag(q)−qqᵀ。"""
    params = list(model.parameters())
    n = sum(p.numel() for p in params)
    with sdpa_kernel([SDPBackend.MATH]):
        logits, _ = model(x, y)
    q = F.softmax(logits, dim=-1)
    B, T, V = logits.shape
    flat = logits.reshape(-1, V)
    J = []
    for i in range(flat.shape[0]):
        for v in range(V):
            r = torch.autograd.grad(flat[i, v], params, retain_graph=True, allow_unused=True)
            J.append(torch.cat([
                (torch.zeros_like(p).reshape(-1) if t is None else t.reshape(-1))
                for t, p in zip(r, params)]))
    J = torch.stack(J).reshape(B * T, V, n)
    G = torch.zeros(n, n, dtype=J.dtype)
    qf = q.reshape(-1, V)
    for t in range(B * T):
        A = torch.diag(qf[t]) - torch.outer(qf[t], qf[t])
        G += J[t].T @ A @ J[t]
    return G / (B * T)


def flat_index(model, block):
    """块坐标 → 全局 flat 索引（用于和暴力矩阵对齐）。"""
    off, base = {}, 0
    for nm, p in model.named_parameters():
        off[nm] = base
        base += p.numel()
    idx = []
    for s in block.specs:
        idx.extend(range(off[s.param] + s.start, off[s.param] + s.stop))
    return torch.tensor(idx)


def main():
    torch.manual_seed(0)
    # 极小模型：暴力 Hessian 是 O(n²) 且 GN 要 T·V 次反向，n 必须很小
    cfg = TransformerConfig(D=8, L=2, M=8, H=2, K=4, V=16, seq_len=4)
    model = Transformer(cfg).double().eval()

    # ⚠ 必须打破 head 的零初始化（model.reset_parameters 里 self.head.zero_()）。
    # head=0 → logits≡0 → ∂logits/∂embd = (path)×head = 0 → H_embd,embd 恒为**零矩阵**，
    # 那样对拍就是拿 0 和 0 比，测不出任何东西（首版就栽在这）。
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.5)
    n = model.n_params()
    x = torch.randint(0, cfg.V, (1, cfg.seq_len))
    y = torch.randint(0, cfg.V, (1, cfg.seq_len))
    print(f"n_params = {n}")

    blocks = build_layers(model, "embd,head,layer00,layer01")
    for b in blocks:
        print(" ", b)

    cov, tot, ov = check_partition(model, blocks)
    print(f"覆盖 {cov}/{tot}  重叠 {ov}   (应为完全覆盖、零重叠)")

    for kind, brute_fn in (("hessian", brute_hessian), ("gn", brute_gn)):
        print(f"\n=== {kind} ===")
        Mfull = brute_fn(model, x, y).detach()
        # ⚠ 判据阈值不能取 double 的 1e-15：model.py 的 rmsnorm/apply_rope 内部硬转
        # .float()（行 51/62），所以即便模型是 double，计算精度也只有 fp32。
        # 用暴力矩阵自身的不对称度作为噪声底，判据取其 100 倍。
        asym = float((Mfull - Mfull.T).abs().max())
        tol = max(asym * 100, 1e-6)
        print(f"  暴力矩阵不对称度 {asym:.2e}（fp32 噪声底）→ 判据 tol={tol:.1e}")

        # (1)(2) 逐块对拍
        vs, worst = [], 0.0
        for b in blocks:
            g = torch.Generator().manual_seed(block_seed(0, b.name))
            vs.append(torch.randn(b.numel, generator=g, dtype=torch.float64))
        op = SharedBatchOp(model, x, y, kind, None)
        op.prepare(blocks)                      # lockstep：一次 prepare 服务全部块
        got_lockstep = [op.apply(b, v) for b, v in zip(blocks, vs)]
        op.release()

        for b, v, got in zip(blocks, vs, got_lockstep):
            idx = flat_index(model, b)
            want = Mfull[idx][:, idx] @ v
            wn = float(want.norm())
            assert wn > 1e-12, f"{b.name} 的块 Hessian 近乎为零，对拍无意义"
            rel = float((got - want).norm()) / wn
            worst = max(worst, rel)
            print(f"  {b.name:16s} n_b={b.numel:6d}  ‖want‖={wn:.3e}  rel_err={rel:.3e}")
        print(f"  最差相对误差 {worst:.3e}  → {'✅' if worst < tol else '❌'}")

        # (3) lockstep vs 逐块独立 prepare
        d = 0.0
        for b, v, got in zip(blocks, vs, got_lockstep):
            op1 = SharedBatchOp(model, x, y, kind, None)
            op1.prepare([b])
            solo = op1.apply(b, v)
            op1.release()
            d = max(d, float((got - solo).abs().max()))
        print(f"  lockstep vs 单块 最大逐位差 {d:.3e}  → {'✅' if d == 0.0 else '⚠'}")

        # full block 应等于整个矩阵
        fb = full_block(model)
        vf = torch.randn(n, dtype=torch.float64)
        op2 = SharedBatchOp(model, x, y, kind, None)
        op2.prepare([fb]); gotf = op2.apply(fb, vf); op2.release()
        relf = float((gotf - Mfull @ vf).norm() / (Mfull @ vf).norm())
        print(f"  full block rel_err={relf:.3e}  → {'✅' if relf < tol else '❌'}")

        # (4) (M²)_bb vs (M_bb)²：块谱不可反推的定量证据
        b0 = blocks[0]
        idx = flat_index(model, b0)
        M2_bb = (Mfull @ Mfull)[idx][:, idx]
        Mbb2 = Mfull[idx][:, idx] @ Mfull[idx][:, idx]
        num = float((M2_bb - Mbb2).norm())
        den = float(Mbb2.norm())
        tr2, trb = float(M2_bb.trace()), float(Mbb2.trace())
        print(f"  块 {b0.name}: ‖(M²)_bb − (M_bb)²‖/‖(M_bb)²‖ = {num/den:.3f}"
              f"   tr(M²)_bb={tr2:.4e} ≥ tr(M_bb²)={trb:.4e}"
              f"  {'✅' if tr2 >= trb - 1e-12 else '❌'}")

        # (5) 仅 GN：dense_block_eig 的 lm_head 闭式块 vs 暴力矩阵
        #     按 v 切：G_{vv} = (1/T) Σ_t q_tv(1−q_tv) x_t x_tᵀ
        #     head 是最后一个参数、形状 (D,V)，vec 索引序 (d,v) → 列 v 的坐标是
        #     head 段内每隔 V 取一个（stride=V, offset=v）。
        if kind == "gn":
            from dense_block_eig import head_col_spectra
            from src.model.model import rmsnorm as _rn
            with torch.no_grad():
                h = model.embd[x]
                for l in range(cfg.L):
                    h = model._block(h, l)
                hn = _rn(h, cfg.norm_eps)
                lg = torch.einsum("...TD,DV->...TV", hn, model.head)
                Xc = hn.reshape(-1, cfg.D).double().cpu()
                Qc = F.softmax(lg.double(), dim=-1).reshape(-1, cfg.V).cpu()
            base = 0
            for nm, p_ in model.named_parameters():
                if nm == "head":
                    break
                base += p_.numel()
            worst_cf = 0.0
            for v0 in (0, cfg.V // 2, cfg.V - 1):
                ev_cf = head_col_spectra(Xc, Qc, [v0], torch.device("cpu"))[v0]
                cidx = torch.tensor([base + d * cfg.V + v0 for d in range(cfg.D)])
                ev_bf = torch.linalg.eigvalsh(
                    Mfull[cidx][:, cidx].detach()).numpy()
                rel = np.abs(np.sort(ev_cf) - np.sort(ev_bf)).max() / max(
                    np.abs(ev_bf).max(), 1e-30)
                worst_cf = max(worst_cf, float(rel))
            print(f"  lm_head 闭式(按 v 切) vs 暴力 最差谱误差 {worst_cf:.3e}"
                  f"  → {'✅' if worst_cf < tol else '❌'}")

        # (6) 预条件路径：对拍暴力 (P H P)_bb。
        #     必须同时覆盖两种预条件器形状（见 spectrum_ddp.load_checkpoint）：
        #       raw  = 0 维标量（CompleteP 每张量一个乘子）
        #       adam = 与参数同形的张量
        #     首版 _precond_block 无脑 .reshape(-1)[start:stop]，把标量切成空张量 →
        #     RuntimeError: size of tensor a (1048576) must match tensor b (0)。
        for plabel, mk in (
            ("raw(0维标量)", lambda p: torch.tensor(0.7, dtype=torch.float64)),
            ("adam(同形张量)", lambda p: torch.rand_like(p) + 0.5),
        ):
            pre = {nm: mk(p_) for nm, p_ in model.named_parameters()}
            dvec = torch.cat([(pre[nm] * torch.ones_like(p_)).reshape(-1)
                              for nm, p_ in model.named_parameters()])
            PMP = torch.diag(dvec) @ Mfull @ torch.diag(dvec)
            worst_p = 0.0
            for b in blocks:
                vv = torch.randn(b.numel, dtype=torch.float64)
                opp = SharedBatchOp(model, x, y, kind, pre)
                opp.prepare([b])
                gotp = opp.apply(b, vv)
                opp.release()
                ii = flat_index(model, b)
                wantp = PMP[ii][:, ii] @ vv
                worst_p = max(worst_p, float((gotp - wantp).norm() / wantp.norm()))
            print(f"  预条件 {plabel:14s} 最差相对误差 {worst_p:.3e}"
                  f"  → {'✅' if worst_p < tol else '❌'}")


if __name__ == "__main__":
    main()
