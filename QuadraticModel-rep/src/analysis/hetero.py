"""
谱异质性度量（从 toy_models/hessian_toy.py 原样搬来，保持口径一致）。

把每个 unit 的特征值谱转成 log 空间概率直方图，再两两算 Symmetric-KL / JS 距离。
用于「层内 unit 间异质性」热图（analyze_blocks → plot_blocks）。
"""
import numpy as np
import torch

EPS = 1e-12


def spectra_to_prob(eig_rows, edges):
    """eig_rows: (n_units, k) 特征值。返回 (n_units, n_bins) 概率行，
    每行是该 unit 特征值在 log 空间的归一化直方图。"""
    P = []
    for row in eig_rows:
        vals = np.clip(np.asarray(row, float), 0.0, None)
        logs = np.log(vals + EPS)
        hist, _ = np.histogram(logs, bins=edges)
        hist = hist.astype(np.float64) + EPS          # Dirichlet 平滑
        P.append(hist / hist.sum())
    return np.vstack(P)


def common_log_edges(eig_rows, num_bins=64):
    allv = np.clip(np.concatenate([np.asarray(r, float).ravel() for r in eig_rows]), 0.0, None)
    z = np.log(allv + EPS)
    zmin, zmax = float(z.min()), float(z.max())
    if zmin == zmax:
        zmin, zmax = zmin - 1e-6, zmax + 1e-6
    return np.linspace(zmin, zmax, num_bins + 1)


def pairwise_matrix(P, metric, device="cpu", chunk=64):
    """P 的行两两 Symmetric-KL ('skl') 或 JS 距离 ('js') 矩阵，(n,n)，对角为 0。"""
    Pt = torch.as_tensor(np.clip(P, EPS, None), dtype=torch.float64, device=device)
    Pt = Pt / Pt.sum(dim=1, keepdim=True)
    logP = torch.log(Pt)
    n = Pt.shape[0]
    if metric == "skl":
        a = (Pt * logP).sum(dim=1)
        M = Pt @ logP.t()
        D = (a[:, None] + a[None, :]) - (M + M.t())
    else:
        D = torch.zeros((n, n), dtype=torch.float64, device=device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            Pi = Pt[s:e][:, None, :]; Li = logP[s:e][:, None, :]
            Pj = Pt[None, :, :];      Lj = logP[None, :, :]
            m = 0.5 * (Pi + Pj); logm = torch.log(m)
            js = 0.5 * (Pi * (Li - logm)).sum(-1) + 0.5 * (Pj * (Lj - logm)).sum(-1)
            D[s:e] = torch.sqrt(js.clamp_min(0.0))
    D.fill_diagonal_(0.0)
    return D.cpu().numpy()


def hetero_mean(D):
    n = D.shape[0]
    idx = np.tril_indices(n, k=-1)
    return float(D[idx].mean()) if len(idx[0]) else 0.0
