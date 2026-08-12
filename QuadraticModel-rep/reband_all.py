"""
从主 npz 的 (eigs, weights) 无损重建 Lanczos alpha/beta，用 gauss_radau 修好的
残差判据 L + 连续体节点间距误差带重算每条曲线，写出 *_reband.npz（含 alpha/beta）。
四条曲线：gn_adam / hessian_adam / gn_sgd / hessian_sgd。
"""
import numpy as np
from rebuild_bands import lanczos_from_measure
from gauss_radau import compute_spectrum_with_error_bands

SRC = "outputs/spectrum_ddp_p100_m1200.npz"
DST = "outputs/spectrum_ddp_p100_m1200_reband.npz"
CURVES = ["gn_adam", "hessian_adam", "gn_sgd", "hessian_sgd"]

z = np.load(SRC, allow_pickle=True)
N = int(np.atleast_1d(z["n_params"])[0])
out = {k: z[k] for k in ["m", "n_params", "n_tokens", "ema"] if k in z.files}

for c in CURVES:
    eigs, wts = z[f"{c}_eigs"], z[f"{c}_weights"]
    alpha, beta = lanczos_from_measure(eigs, wts)
    spec = compute_spectrum_with_error_bands(alpha, beta, n_params=N, n_grid=400)
    for k, v in spec.items():
        out[f"{c}_{k}"] = v
    out[f"{c}_eigs"], out[f"{c}_weights"] = eigs, wts
    out[f"{c}_alpha"], out[f"{c}_beta"] = alpha, beta
    print(f"{c:14s} L={spec['L']:4d} cut={spec['cut']:4d} "
          f"lam_max={spec['y'].max():.3f} lam_min={spec['y'].min():.3e}")

np.savez(DST, **out)
print("saved:", DST)
