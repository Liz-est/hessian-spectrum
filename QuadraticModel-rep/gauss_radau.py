"""
Gauss-Radau Quadrature 用于 Lanczos 谱的误差带估计

对齐论文原版 QuadraticModel/analysis/spectrum/{postprocess.py,quadrature.py}
的 hessian/gn 分支（scipy-only 移植）。理论基础（Golub & Meurant, "Matrices,
Moments and Quadrature and their applications"，Chapter 6）：
- Lanczos 三对角矩阵 T_m 的特征分解给出 m 点 Gauss 规则，节点=Ritz 值 θ_i、
  权重=U[0,i]^2，近似谱测度 μ。累积分布 Φ(t)=μ({λ≤t})，绘图横轴
  eigenvalue index = N_params × (1 − Φ(t))。
- 误差带：把 Gauss-Radau 规则的固定节点逐格点锚定在 t 上（quadrature.py 的
  radau_batch），得到 index 的一对上下界；中线由 band_midpoint 求。
- hessian 分支 cut=m、跨零 signed-log 网格覆盖 [λmin,λmax]、R>0；gn（半正定）
  分支 cut=#positive、R=0、网格只在正谱。详见 compute_spectrum_with_error_bands。
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal


# ============================================================================
# 论文原版 postprocess（analysis/spectrum/postprocess.py + quadrature.py）
# 的 hessian 分支 scipy-only 移植。旧的启发式误差带（node-gap gradient、
# cut=#positive、R=0）会把整条尾巴截断在「最小正 Ritz 节点」——而 167M 参数
# 的 Hessian 只有几百个非平凡特征值，其余零空间被 Lanczos 塌缩成一个位于机器零
# 附近的 Ritz 值，它的符号(±1e-7)由舍入误差决定。旧逻辑用 cut=#positive，让整条
# 尾巴长度取决于这个无意义的符号（1M 落在 +3.99e-7 → 尾到 1.63e8；5M 落在
# -6.89e-7 → 被丢弃、尾只到 1.30e7）。原版用 cut=m + 跨零 signed-log 网格 +
# 逐格点 Gauss-Radau 锚定(R>0)，两条曲线自洽且都延伸到 N。
# ============================================================================


def _signed_log(x, t):
    return np.sign(x) * np.log1p(np.abs(x) / t)


def _signed_exp(x, t):
    return np.sign(x) * t * np.expm1(np.abs(x))


def _evaluation_grid(low, high, size, threshold):
    """跨零 signed-log 网格（原版 postprocess.evaluation_grid）。"""
    eps = 3e-8
    if low > 0 and high > 0:
        return np.geomspace(low * (1 + eps), high * (1 - eps), size)
    if low < 0 and high < 0:
        return -np.geomspace(abs(low) * (1 - eps), abs(high) * (1 + eps), size)
    tl, th = _signed_log(np.asarray([low, high]), threshold)
    margin = eps * (th - tl)
    return _signed_exp(np.linspace(tl + margin, th - margin, size), threshold)


def _nudge_from_ritz(value, ritz, rd=3e-8):
    """把落在 Ritz 值上的格点微推离开，避免 Radau 固定节点退化。"""
    i = np.argmin(np.abs(ritz - value))
    e = rd * max(abs(ritz[i]), abs(value), np.finfo(float).tiny)
    if abs(value - ritz[i]) < e:
        side = np.sign(value - ritz[i]) or -1.0
        value = ritz[i] + side * e
    return value


def _weights_from_eigenvalues(theta, diagonal, off_diagonal):
    """三项递推稳定地算 Radau 规则权重（原版 quadrature 同名函数）。"""
    xp = np.ones_like(theta)
    x = (theta - diagonal[0]) / off_diagonal[0]
    total = xp * xp + x * x
    nrm = np.maximum(1.0, np.maximum(np.abs(xp), np.abs(x)))
    xp /= nrm
    x /= nrm
    total /= nrm * nrm
    log_scale = np.log(nrm)
    for i in range(1, len(diagonal) - 1):
        xn = ((theta - diagonal[i]) * x - off_diagonal[i - 1] * xp) / off_diagonal[i]
        nrm = np.maximum(1.0, np.maximum(np.abs(x), np.abs(xn)))
        total = (total + xn * xn) / (nrm * nrm)
        xp = x / nrm
        x = xn / nrm
        log_scale += np.log(nrm)
    return np.exp(-np.log(total) - 2 * log_scale)


def _radau_batch_scipy(a, b, lams, left_locked, right_locked, scale):
    """逐格点把 Gauss-Radau 固定节点锚在 lam 上，返回 (above, above+forced) 的
    index 上下界对（原版 quadrature.radau_batch_scipy）。b 长度为 m（末位=残差 β_m）。"""
    off = b[: len(a) - 1]
    beta = b[len(a) - 1]
    d = a[0] - lams
    for i in range(1, len(a)):
        d = a[i] - lams - off[i - 1] ** 2 / d
    eoff = np.r_[off, beta]
    result = []
    for lam, last_alpha in zip(lams, lams + beta ** 2 / d):
        diagonal = np.r_[a, last_alpha]
        theta = eigh_tridiagonal(
            diagonal, eoff, eigvals_only=True,
            check_finite=False, lapack_driver="sterf",
        )
        w = scale * _weights_from_eigenvalues(theta, diagonal, eoff)
        if left_locked or right_locked:
            locked = np.zeros_like(theta, dtype=bool)
            if right_locked:
                locked[:right_locked] = True
            if left_locked:
                locked[-left_locked:] = True
            w[locked] = 1.0
            w[~locked] *= (scale - left_locked - right_locked) / w[~locked].sum()
        fi = np.argmin(np.abs(theta - lam))
        nf = np.ones_like(theta, dtype=bool)
        nf[fi] = False
        above = w[nf & (theta > lam)].sum()
        result.append((above, above + w[fi]))
    return np.asarray(result)


def _distance_to_polyline_squared(x, y, px, py):
    x0, y0 = px[:-1][None], py[:-1][None]
    dx = (px[1:] - px[:-1])[None]
    dy = (py[1:] - py[:-1])[None]
    x, y = x[:, None], y[:, None]
    weight = np.clip(((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.min((x - x0 - weight * dx) ** 2 + (y - y0 - weight * dy) ** 2, axis=1)


def _band_midpoint(values, lower, upper, threshold):
    """原版 band_midpoint：在 log-index / signed-log-eig 空间用二分求 lo/hi 中线。"""
    lower_log = np.log(lower)
    upper_log = np.log(upper)
    values_log = _signed_log(values, threshold)
    left, right = lower_log.copy(), upper_log.copy()
    for _ in range(48):
        center = 0.5 * (left + right)
        closer = _distance_to_polyline_squared(
            center, values_log, lower_log, values_log
        ) < _distance_to_polyline_squared(center, values_log, upper_log, values_log)
        left = np.where(closer, center, left)
        right = np.where(closer, right, center)
    return np.exp(0.5 * (left + right))


def compute_spectrum_with_error_bands(
    alpha: np.ndarray,
    beta: np.ndarray,
    n_params: int,
    n_grid: int = 400,
    curvature: str = "hessian",
    rel_tol: float = 1e-6,
    linthresh: float = 1e-8,
    radau_chunk: int = 32,
    conv_tol: float | None = None,   # 兼容旧调用签名（已弃用，不再使用）
) -> dict:
    """
    Lanczos α/β → 谱曲线，对齐论文原版 postprocess.spectrum_curve 的 hessian/gn 分支。

    - curvature="hessian"（默认，裸 Hessian / 预条件 Hessian）：cut=m，跨零 signed-log
      网格覆盖 [λmin, λmax]，逐格点 Gauss-Radau 锚定给出 index 上下界(R>0)。
    - curvature="gn"（半正定 Gauss-Newton）：R=0，cut=#positive，网格只在正谱。

    收敛判据用残差 rel_i = β_m·|U[-1,i]|/|θ_i| < rel_tol（原版默认 1e-6）划出左端已锁定
    离散谱 L 与右端已锁定 R；其余为连续体，rank 用 √(lower·upper) 质量分配。

    Args:
        alpha: (m,) 三对角对角元素
        beta:  (m-1,) 次对角，或 (m,) 末位为残差 β_m。长度 m-1 时用 β_{m-1} 作残差代理
               （已验证正谱尾对残差在 [0.5×,2×] 区间不敏感）。
        n_params: 参数总数 N（index 缩放）
        n_grid: 网格点数
        curvature: "hessian" | "gn"

    Returns:
        dict(x, y, dot_x, g, lo, hi, mid, L, R, cut)：x/mid/lo/hi 是 eigenvalue index，
        y/g 是 eigenvalue。plotter combined_positive 用 x[:L]/y[:L] 画锁定头、mid/g 画连续体。
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    m = len(alpha)
    # 统一成原版约定：betas 长度 m，末位是残差 β_m
    if len(beta) == m - 1:
        betas = np.r_[beta, beta[-1]]     # 残差未保存 → 用 β_{m-1} 作稳定代理
    elif len(beta) == m:
        betas = beta
    else:
        raise ValueError(f"beta length {len(beta)} incompatible with alpha length {m}")

    N = float(n_params)
    ev, U = eigh_tridiagonal(alpha, betas[:-1])
    ev = ev[::-1]
    U = U[:, ::-1]
    weights = U[0] ** 2
    residuals = betas[-1] * np.abs(U[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        locked = residuals / np.abs(ev) < rel_tol

    left_locked = 0
    while left_locked < len(ev) and locked[left_locked]:
        left_locked += 1
    right_locked = 0
    while (right_locked < len(ev) - left_locked
           and locked[len(ev) - 1 - right_locked]):
        right_locked += 1

    positive_semidefinite = curvature == "gn"
    if positive_semidefinite:
        right_locked = 0
        cut = int((ev > 0).sum())
    else:
        cut = len(ev)

    # rank 权重：锁定端 index=整数序号，连续体按 √(lower·upper) 质量分配
    rank_weights = N * weights / weights.sum()
    if left_locked:
        rank_weights[:left_locked] = 1
    if right_locked:
        rank_weights[-right_locked:] = 1
    if left_locked + right_locked < len(rank_weights):
        stop = len(rank_weights) - right_locked
        rank_weights[left_locked:stop] *= (
            N - left_locked - right_locked
        ) / rank_weights[left_locked:stop].sum()

    ranks = np.empty_like(ev)
    if left_locked:
        ranks[:left_locked] = np.arange(1, left_locked + 1)
    if right_locked:
        ranks[-right_locked:] = N - right_locked + np.arange(1, right_locked + 1)
    if left_locked + right_locked < len(ranks):
        stop = len(ranks) - right_locked
        edges = np.r_[0, np.cumsum(rank_weights[left_locked:stop])]
        lower_rank = left_locked + 0.5 + edges[:-1]
        upper_rank = left_locked + 0.5 + edges[1:]
        ranks[left_locked:stop] = np.sqrt(
            np.maximum(lower_rank, 1e-300) * np.maximum(upper_rank, 1e-300)
        )

    result = {
        "x": ranks,
        "y": ev,
        "dot_x": ranks,
        "L": int(left_locked),
        "R": int(right_locked),
        "cut": int(cut),
    }

    if left_locked < cut:
        low = 0.0 if positive_semidefinite else ev[cut - 1]
        high = ev[left_locked - 1] if left_locked else ev[0]
        if high > low:
            grid = _evaluation_grid(low, high, n_grid, linthresh)
            grid = np.asarray([_nudge_from_ritz(v, ev) for v in grid])
            bounds = []
            for start in range(0, len(grid), radau_chunk):
                bounds.append(_radau_batch_scipy(
                    alpha, betas, grid[start:start + radau_chunk],
                    left_locked, right_locked, N,
                ))
            bounds = np.concatenate(bounds)
            lower = np.maximum.accumulate(bounds[:, 0][::-1])[::-1]
            upper = np.minimum.accumulate(bounds[:, 1])
            lower = np.maximum(lower, 1e-12)
            upper = np.maximum(upper, lower)
            midpoint = _band_midpoint(grid, lower, upper, linthresh)
            result.update(g=grid, lo=lower, hi=upper, mid=midpoint)

    result.setdefault("g", np.asarray([]))
    result.setdefault("lo", np.asarray([]))
    result.setdefault("hi", np.asarray([]))
    result.setdefault("mid", np.asarray([]))
    return result


# ============================================================================
# 测试函数
# ============================================================================

def test_gauss_radau():
    """用已知对称矩阵测试：Lanczos 累积密度 vs 真实累积分布，并检查带序。"""
    print("=" * 80)
    print("测试 Gauss-Radau Quadrature（真实 alpha/beta，端点锚定）")
    print("=" * 80)

    np.random.seed(0)
    n = 300
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    true_eigs = np.sort(np.abs(np.random.randn(n)) * 5 + 0.1)
    A = Q @ np.diag(true_eigs) @ Q.T

    # Lanczos（完整重正交）生成 alpha/beta
    m = 80
    v = np.random.randn(n); v /= np.linalg.norm(v)
    V = np.zeros((m, n)); alpha = np.zeros(m); beta = np.zeros(m - 1)
    V[0] = v
    w = A @ v; alpha[0] = w @ v; w = w - alpha[0] * v
    for j in range(1, m):
        for i in range(j):
            w = w - (w @ V[i]) * V[i]
        b = np.linalg.norm(w); beta[j - 1] = b; V[j] = w / b
        w = A @ V[j]; alpha[j] = w @ V[j]
        w = w - alpha[j] * V[j] - beta[j - 1] * V[j - 1]

    print(f"矩阵 {n}×{n}, Lanczos m={m}")
    print(f"真实特征值范围: [{true_eigs.min():.4f}, {true_eigs.max():.4f}]")

    result = compute_spectrum_with_error_bands(alpha, beta, n_params=n, n_grid=200)

    print(f"\n网格 g 范围: [{result['g'].min():.4f}, {result['g'].max():.4f}]")
    print(f"mid index 范围: [{result['mid'].min():.2f}, {result['mid'].max():.2f}]")
    print(f"lo  index 范围: [{result['lo'].min():.2f}, {result['lo'].max():.2f}]")
    print(f"hi  index 范围: [{result['hi'].min():.2f}, {result['hi'].max():.2f}]")

    order_ok = np.all(result['lo'] <= result['mid'] + 1e-6) and \
               np.all(result['mid'] <= result['hi'] + 1e-6)

    # 真实 index 曲线：index(t) = n × P(eig ≥ t)
    test_lam = np.median(true_eigs)
    true_index = np.sum(true_eigs >= test_lam)
    est_index = np.interp(test_lam, result['g'], result['mid'])
    print(f"\n在 λ={test_lam:.3f} 处 (index = #{{eig ≥ λ}}):")
    print(f"  真实 index: {true_index}")
    print(f"  估计 index: {est_index:.2f}")
    print(f"  相对误差: {abs(true_index - est_index) / max(true_index,1):.4f}")

    if order_ok:
        print("\n✅ 测试通过！误差带满足 lo ≤ mid ≤ hi")
    else:
        bad = np.sum((result['lo'] > result['mid'] + 1e-6) |
                     (result['mid'] > result['hi'] + 1e-6))
        print(f"\n⚠️  {bad}/{len(result['mid'])} 个网格点违反带序")
    print("=" * 80)


if __name__ == "__main__":
    test_gauss_radau()
