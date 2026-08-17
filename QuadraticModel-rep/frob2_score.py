"""
逐序列 Gauss-Newton Frobenius-norm² 打分（Hutchinson 探针），复现论文
preprocessing/sample_hessian_frob2.py 的 frob2 过滤所用分数。

数学：单条序列 i 的 GN 矩阵 G_i = (1/T)Σ_t J_tᵀ A_t J_t（A_t 是 softmax 曲率
diag(q)-qqᵀ）。它 167M×167M 无法显式装配，但

    ‖G_i‖_F² = tr(G_iᵀG_i) = E_v[‖G_i v‖²],   v ~ Rademacher(±1) 且与数据独立

用 num_probes 个独立 Rademacher 探针的 ‖G_i v‖² 均值无偏估计 ‖G_i‖_F²。
探针 v 是与序列无关的随机「尺子」，独立性是 Hutchinson 恒等式成立的前提。

预条件：论文的 frob2 过滤器**只有一把**，用 **Adam 预条件的 GN**（
sample_hessian_frob2.py 默认 include_adam_precond=True）。这里直接把 spectrum 的
Adam precond dict 传给 gauss_newton_vector_product —— 它左右各乘 P，返回 M v = P·G_i·P·v，
于是估到的是 ‖P G_i P‖_F²（Adam 预条件 GN 的 frob2），与论文一致。四条谱曲线（gn/hessian
× adam/raw）共用这**同一份**打分切出的保留子集（见 analysis/spectrum/run_basis.py:26
REST_CSV_REL 写死指向 filtered_data.csv，不随曲线的 preconditioner 变）。

本模块与分布式无关：只提供逐序列打分。序列如何分给各 rank、分数如何 all_gather，
由调用方（spectrum_frob2_ddp.py）处理。
"""
from __future__ import annotations

import torch

from hvp import gauss_newton_vector_product


def _rademacher(n, generator, device):
    """±1 Rademacher 向量 (n,) float32。"""
    bits = torch.randint(0, 2, (n,), generator=generator, device=device, dtype=torch.int8)
    return bits.to(torch.float32).mul_(2.0).sub_(1.0)


@torch.no_grad()
def _noop():
    # 占位：本文件不需要 no_grad 包装（GN-VP 内部自建图），保留以示意 forward 语义。
    pass


def score_one_sequence(model, x, y, precond, num_probes, seed_base, device, n_params):
    """单条序列的 Adam-precond GN frob2 估计（num_probes 个 Rademacher 探针均值）。

    Args:
        model: Transformer（eval 态）
        x, y: (1, seq_len) int64，单条序列的输入/目标
        precond: Adam 预条件 dict {param_name: √(pre·post/(√ν̂+eps))}
        num_probes: 探针数
        seed_base: 该序列探针 RNG 的基种子（不同序列取不同 base 保证探针独立可复现）
        device: 计算设备
        n_params: 参数总维度

    Returns:
        float: mean_p ‖P G P v_p‖²  ≈ ‖P G P‖_F²
    """
    gen = torch.Generator(device=device)
    acc = 0.0
    for p in range(num_probes):
        gen.manual_seed(seed_base + p)
        v = _rademacher(n_params, gen, device)
        mv = gauss_newton_vector_product(model, x, y, v, preconditioner=precond)
        acc += float(torch.dot(mv, mv).item())
    return acc / num_probes


def score_sequences(model, seqs, precond, num_probes, probe_seed, device, n_params,
                    idxs=None, log_every=50, logfn=None):
    """给一批序列打分。

    Args:
        seqs: list[(x, y)]，每个 (1, seq_len)，**全局**序列池（所有 rank 相同）
        idxs: 要打分的全局下标列表（本 rank 负责的子集）；None=全部
        probe_seed: 全局探针种子基；序列 g 的探针 base = probe_seed + g*num_probes
                    （用全局下标 g 保证与 rank 划分无关、可复现）
    Returns:
        dict {global_idx: score}
    """
    if idxs is None:
        idxs = list(range(len(seqs)))
    out = {}
    for k, g in enumerate(idxs):
        x, y = seqs[g]
        base = probe_seed + g * num_probes
        out[g] = score_one_sequence(model, x, y, precond, num_probes, base, device, n_params)
        if logfn is not None and (k + 1) % log_every == 0:
            logfn(f"  frob2 打分 {k+1}/{len(idxs)}  (global#{g} = {out[g]:.4e})")
    return out
