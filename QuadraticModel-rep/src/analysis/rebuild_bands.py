"""
从 npz 里的 (eigs, weights) 无损重建 Lanczos 三对角 alpha/beta，
再用不同 Gauss-Radau 锚点重算误差带，定位「带子塌成零宽」是不是锚点选错。

重建原理：离散测度 μ = Σ_i w_i δ(λ_i) 的正交多项式三项递推系数 (alpha,beta)
= 对 A=diag(eigs)、起始向量 v0=sqrt(weights) 跑 Lanczos 得到的三对角。
其 Gauss 规则精确等于 (eigs,weights)。带完整重正交。
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src import paths
from src.spectrum.gauss_radau import _radau_matrix, _cdf_from_rule, cumulative_spectral_density


def lanczos_from_measure(nodes, weights, reorth=True):
    """对 diag(nodes) 以 sqrt(weights) 为起点跑 Lanczos，恢复 alpha/beta。"""
    x = np.asarray(nodes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    m = len(x)
    v0 = np.sqrt(w)
    nrm = np.linalg.norm(v0)
    if nrm == 0:
        raise ValueError("zero weight vector")
    v0 = v0 / nrm
    Q = np.zeros((m, m))
    alpha = np.zeros(m)
    beta = np.zeros(m - 1)
    q_prev = np.zeros(m)
    q = v0
    b_prev = 0.0
    for j in range(m):
        Q[:, j] = q
        Aq = x * q
        a = float(q @ Aq)
        alpha[j] = a
        r = Aq - a * q - b_prev * q_prev
        if reorth:
            r = r - Q[:, :j + 1] @ (Q[:, :j + 1].T @ r)
        b = np.linalg.norm(r)
        if j < m - 1:
            beta[j] = b
            if b < 1e-300:
                break
            q_prev = q
            q = r / b
            b_prev = b
    return alpha, beta


def bands(alpha, beta, n_params, grid, anchor_lo, anchor_hi):
    nodes, weights = cumulative_spectral_density(alpha, beta)
    cdf_gauss = _cdf_from_rule(nodes, weights, grid)
    try:
        a, b = _radau_matrix(alpha, beta, anchor_lo)
        nd, wd = cumulative_spectral_density(a, b)
        cdf_a = _cdf_from_rule(nd, wd, grid)
    except Exception:
        cdf_a = cdf_gauss
    try:
        a, b = _radau_matrix(alpha, beta, anchor_hi)
        nd, wd = cumulative_spectral_density(a, b)
        cdf_b = _cdf_from_rule(nd, wd, grid)
    except Exception:
        cdf_b = cdf_gauss
    cdf_low = np.clip(np.minimum.reduce([cdf_gauss, cdf_a, cdf_b]), 0, 1)
    cdf_high = np.clip(np.maximum.reduce([cdf_gauss, cdf_a, cdf_b]), 0, 1)
    lo = n_params * (1.0 - cdf_high)
    hi = n_params * (1.0 - cdf_low)
    return lo, hi


def relwidth(lo, hi, mid):
    q = (mid > 0)
    return float(np.nanmedian((hi[q] - lo[q]) / np.maximum(mid[q], 1e-30)))


if __name__ == "__main__":
    z = np.load(paths.OUT_DIR / "spectrum_ddp_p100_m1200.npz", allow_pickle=True)
    N = int(np.atleast_1d(z["n_params"])[0])
    for c in ["gn_adam", "gn_sgd", "hessian_adam"]:
        eigs = z[f"{c}_eigs"]; wts = z[f"{c}_weights"]
        g = z[f"{c}_g"]; mid = z[f"{c}_mid"]
        alpha, beta = lanczos_from_measure(eigs, wts)
        nd, _ = cumulative_spectral_density(alpha, beta)
        lmin, lmax = float(nd.min()), float(nd.max())
        print(f"\n=== {c} ===  reconstructed lmin={lmin:.4g} lmax={lmax:.4g}  "
              f"(orig eigs min={eigs.min():.4g} max={eigs.max():.4g})")
        # A) reproduce our narrow band (anchor at Ritz extremes)
        loA, hiA = bands(alpha, beta, N, g, lmin, lmax)
        # B) lower anchor at 0 (GN PSD prior); upper relaxed
        loB, hiB = bands(alpha, beta, N, g, 0.0, lmax * 1.001)
        print(f"  A ritz-extreme anchors : relwidth={relwidth(loA,hiA,mid):.3e}")
        print(f"  B anchor_lo=0          : relwidth={relwidth(loB,hiB,mid):.3e}")
