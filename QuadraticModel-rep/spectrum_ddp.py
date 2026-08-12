"""
分布式 Lanczos 谱计算（16 卡 = 2×8 H100），单 checkpoint 4 曲线。

与单卡 compute_spectrum.py 的两处并行改造：

1. HVP batch 并行：每 rank 只算 1/world 的 batch，all_reduce 求和后除以总 batch。
   HVP 本身需要完整参数向量 v（167M 维），由 rank 广播/all_gather 拼齐。

2. **Lanczos 基向量 Q 按参数维度分片**（关键）：
   m=800 的全量 Q(fp32)=537GB 超单节点 512GB RAM，且全重正交在单 CPU 上会
   变成新瓶颈。因此把参数轴切成 world 份，每 rank 只在 CPU 存自己那片
   Q_local(m × shard)。重正交的 coeff = Σ_rank Q_local·w_local（all_reduce），
   w 的更新各 rank 只动自己那片。内存与重正交都 /world。

   每 Lanczos 步通信：
     - all_gather(w_local_shard) → 完整 w（供下一步 HVP，HVP 要整向量）
     - all_reduce(hvp 局部和)
     - 2× all_reduce(重正交标量系数，DGKS 两轮)
   三对角 α/β 由局部 dot 的 all_reduce 得到，各 rank 完全一致。

逐曲线增量输出：每条曲线算完 rank0 立即把该曲线字段并入 npz 落盘，
方便训练中及时检查（4 条曲线不必等全跑完）。

启动（2 节点 × 8 卡，torchrun 多机）：
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$RANK \
      --master_addr=$MASTER_ADDR --master_port=29500 \
      spectrum_ddp.py --ckpt checkpoints_b64/ckpt_p100.pt \
      --m 800 --n_tokens 1000000 --out outputs/spectrum_ddp_p100_m800.npz

本地 sanity（2 rank CPU gloo）：
  torchrun --standalone --nproc_per_node=2 spectrum_ddp.py --ckpt ... --backend gloo --cpu ...
"""
import os, sys, time, math, argparse
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, TransformerConfig
from hvp import hessian_vector_product, gauss_newton_vector_product
from gauss_radau import compute_spectrum_with_error_bands

DATA_DIR = "/data/250010020/hessian-spectrum/data/fineweb_edu_bpe8192_clean"


# --------------------------------------------------------------------------
# 分布式初始化
# --------------------------------------------------------------------------
def setup_dist(backend):
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    return rank, world


def is_master():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(*a):
    """rank0 专用日志（重要里程碑）"""
    if is_master():
        print(*a, flush=True)


def log_all(*a):
    """所有 rank 都输出（带 rank 前缀，Lanczos 步进度用）"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[rank{rank}]", *a, flush=True)


# --------------------------------------------------------------------------
# 参数分片：把 [0, n_params) 尽量均匀切成 world 段
# --------------------------------------------------------------------------
def shard_bounds(n_params, world):
    """返回长度 world 的 (start, end) 列表，尽量均匀（前若干片多 1）。"""
    base = n_params // world
    rem = n_params % world
    bounds, off = [], 0
    for r in range(world):
        sz = base + (1 if r < rem else 0)
        bounds.append((off, off + sz))
        off += sz
    assert off == n_params
    return bounds


# --------------------------------------------------------------------------
# checkpoint 加载（与 run_spectrum_b64.py 对齐：EMA 参数 + EMA'd ν 预条件器）
# --------------------------------------------------------------------------
def load_checkpoint(path, ema_key, device):
    ck = torch.load(path, map_location="cpu")
    c = ck["config"]
    cfg = TransformerConfig(D=c["D"], L=c["L"], M=c["M"], H=c["H"], K=c["K"],
                            V=c["V"], seq_len=c["seq_len"])
    model = Transformer(cfg)

    ema = ck.get("ema")
    if ema is not None and str(ema_key) in ema:
        sd = model.state_dict()
        for name, v in ema[str(ema_key)].items():
            sd[name] = v
        model.load_state_dict(sd)
        used = f"EMA params ema={ema_key}"
    else:
        model.load_state_dict(ck["model"])
        used = "last-iterate params"
    model.eval().to(device)

    # Adam 预条件器：P = √(pre·post/(√ν̂ + eps))，ν̂ = EMA'd ν / (1 - Πβ2)
    # ⚠ pre/post 从 model.lr_groups() 重建（opt_state 里的 pre/post 有旧序列化 bug）
    opt = ck["opt"]
    log_prod_b2 = ck.get("log_prod_b2", opt.get("log_prod_b2", 0.0))
    denom_b2 = max(-math.expm1(log_prod_b2), 1e-16)
    nu_ema = ck.get("nu_ema")
    if nu_ema is not None and str(ema_key) in nu_ema:
        nu_src = nu_ema[str(ema_key)]
        used_nu = f"EMA'd ν ema={ema_key}"
    else:
        nu_src = opt["nu"]
        used_nu = "last ν"
    eps = opt["eps"]
    lr_groups = {name: (pre, post) for name, _, pre, post in model.lr_groups()}
    precond = {}
    for name, _ in model.named_parameters():
        pre, post = lr_groups[name]
        nu_hat = nu_src[name].to(device).float() / denom_b2
        precond[name] = torch.sqrt(pre * post / (torch.sqrt(nu_hat) + eps))
    log(f"  {used}; {used_nu}")
    return model, cfg, precond


# --------------------------------------------------------------------------
# 数据：每 rank 只采自己那一份 batch（HVP batch 并行）
# --------------------------------------------------------------------------
def make_local_batches(cfg, n_tokens, world, rank, device, seed):
    """
    全局要过 n_tokens，每 rank 分 1/world。用 rank 相关 seed 保证不同 rank 采不同数据。
    返回 (local_batches, global_nbatch)：global_nbatch 用于 HVP 求和后归一化。
    """
    data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
    seq = cfg.seq_len
    n_seqs_global = max(world, n_tokens // seq)
    per = 8  # 每 minibatch 8 条序列，控显存
    # 全局 minibatch 数（向上取到 world 的倍数，便于均分）
    nb_global = max(world, (n_seqs_global + per - 1) // per)
    nb_global = ((nb_global + world - 1) // world) * world
    nb_local = nb_global // world

    g = torch.Generator().manual_seed(seed + 9973 * rank)
    batches = []
    for _ in range(nb_local):
        ix = torch.randint(len(data) - seq - 1, (per,), generator=g)
        x = torch.stack([torch.from_numpy(data[i:i+seq].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i+1:i+1+seq].astype(np.int64)) for i in ix]).to(device)
        batches.append((x, y))
    return batches, nb_global


# --------------------------------------------------------------------------
# 分布式 HVP：每 rank 用本地 batch 算完整 Hv，all_reduce 求和后 /nb_global
# --------------------------------------------------------------------------
def make_dist_hvp(model, local_batches, nb_global, kind, precond, device):
    def hvp(v_full):
        # v_full: 完整 (n_params,) 在 device 上
        acc = torch.zeros_like(v_full)
        for x, y in local_batches:
            if kind == "gn":
                acc += gauss_newton_vector_product(model, x, y, v_full, preconditioner=precond)
            else:
                acc += hessian_vector_product(model, x, y, v_full, preconditioner=precond)
        # 跨 rank 求和（局部 batch 和 → 全局和），再除全局 batch 数
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        return acc / nb_global
    return hvp


# --------------------------------------------------------------------------
# 分片 Lanczos（Algorithm 1，重正交在归一化前）
#   Q_local: (m, shard) 存 CPU；v/w 完整向量在 GPU 上算 HVP，
#   分片只用于存储与重正交的向量运算。
# --------------------------------------------------------------------------
def lanczos_sharded(hvp_fn, n_params, m, world, rank, bounds, device,
                    store_device, seed, dtype=torch.float32):
    s0, s1 = bounds[rank]
    shard = s1 - s0

    # 初始随机向量：所有 rank 用同一 seed 生成**完整** v0 后各取自己片，保证一致
    torch.manual_seed(seed)
    v_full = torch.randn(n_params, dtype=dtype, device=store_device)
    v_full = v_full / v_full.norm()
    v_local = v_full[s0:s1].clone()

    Q = torch.zeros(m, shard, dtype=dtype, device=store_device)
    Q[0] = v_local

    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m - 1, dtype=np.float64)

    # all_gather 用的缓冲（各 rank shard 大小可能差 1，用 list）
    shard_sizes = [e - s for s, e in bounds]

    def gather_full(local_vec_store):
        """把各 rank 的 shard 拼成完整向量（在 device 上，供 HVP）。"""
        local_dev = local_vec_store.to(device)
        parts = [torch.empty(sz, dtype=dtype, device=device) for sz in shard_sizes]
        dist.all_gather(parts, local_dev)
        return torch.cat(parts)

    for j in range(m):
        # w = H · v_j（需要完整 v_j → all_gather Q[j] 片）
        v_full = gather_full(Q[j])
        w_full = hvp_fn(v_full)            # 完整 (n_params,) 在 device
        w = w_full[s0:s1].to(store_device) # 只留自己片
        del v_full, w_full

        # α_j = <w, q_j> 全局：局部 dot 后 all_reduce
        # ⚠ NCCL 只支持 GPU tensor，CPU scalar 需先搬到 device
        a_local = torch.dot(w, Q[j]).to(device)
        dist.all_reduce(a_local, op=dist.ReduceOp.SUM)
        alpha[j] = a_local.item()

        # 全重正交（DGKS 两轮）：coeff = Q_local·w（(j+1,)），all_reduce 求全局
        Qv = Q[: j + 1]
        for _ in range(2):
            coeff = torch.mv(Qv, w).to(device)  # NCCL 需要 GPU tensor
            dist.all_reduce(coeff, op=dist.ReduceOp.SUM)
            w = w - torch.mv(Qv.t(), coeff.to(store_device))

        # β_j = ‖w‖ 全局
        b_local = torch.dot(w, w).to(device)  # NCCL 需要 GPU tensor
        dist.all_reduce(b_local, op=dist.ReduceOp.SUM)
        beta_j = math.sqrt(max(b_local.item(), 0.0))

        if beta_j < 1e-10:
            log(f"    Lanczos early stop at {j+1}/{m} (β={beta_j:.2e})")
            alpha = alpha[:j + 1]; beta = beta[:j]; m = j + 1
            break

        if j < m - 1:
            beta[j] = beta_j
            Q[j + 1] = w / beta_j
        if (j + 1) % 25 == 0 or j == m - 1:
            log_all(f"Lanczos {j+1}/{m}  α={alpha[j]:.3e} β={beta_j:.3e}")

    # 三对角 → 特征分解（各 rank 数据一致，都算，取 rank0 输出）
    T = np.diag(alpha)
    if len(beta) > 0:
        T[np.arange(m - 1), np.arange(1, m)] = beta
        T[np.arange(1, m), np.arange(m - 1)] = beta
    eigvals, U = np.linalg.eigh(T)
    weights = U[0, :] ** 2
    return eigvals, weights, alpha, beta


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--m", type=int, default=800)
    ap.add_argument("--n_tokens", type=int, default=1_000_000, help="全局 HVP 采样 token 数")
    ap.add_argument("--ema", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    ap.add_argument("--cpu", action="store_true", help="强制 CPU（本地 sanity）")
    args = ap.parse_args()

    rank, world = setup_dist(args.backend)
    if args.cpu:
        device = torch.device("cpu")
    else:
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    store_device = torch.device("cpu")  # Q 分片始终存 CPU

    log(f"world={world} backend={args.backend} device={device}")
    log(f"m={args.m}  n_tokens={args.n_tokens}  ema={args.ema}  ckpt={args.ckpt}")

    model, cfg, precond = load_checkpoint(args.ckpt, args.ema, device)
    n_params = model.n_params()
    bounds = shard_bounds(n_params, world)
    s0, s1 = bounds[rank]
    log(f"  n_params={n_params:,}  每 rank shard≈{(s1-s0):,}  "
        f"Q_local(m={args.m},fp32)={args.m*(s1-s0)*4/1e9:.1f}GB/rank")

    local_batches, nb_global = make_local_batches(cfg, args.n_tokens, world, rank, device, args.seed)
    tokens_actual = nb_global * 8 * cfg.seq_len
    log(f"  HVP: nb_global={nb_global} minibatch × 8 × {cfg.seq_len} = {tokens_actual:,} tokens")

    CURVES = [("gn", True, "gn_adam"), ("hessian", True, "hessian_adam"),
              ("gn", False, "gn_sgd"), ("hessian", False, "hessian_sgd")]

    out = {"m": args.m, "n_params": n_params, "n_tokens": tokens_actual, "ema": args.ema}
    if is_master():
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for kind, use_p, tag in CURVES:
        dist.barrier()
        t0 = time.time()
        p = precond if use_p else None
        log(f"\n=== 曲线 {tag} 开始 ===")
        hvp_fn = make_dist_hvp(model, local_batches, nb_global, kind, p, device)
        eigs, weights, alpha, beta = lanczos_sharded(
            hvp_fn, n_params, args.m, world, rank, bounds, device,
            store_device, args.seed)
        dist.barrier()
        dt = time.time() - t0
        log_all(f"曲线 {tag} 完成，耗时 {dt:.0f}s")
        if is_master():
            spec = compute_spectrum_with_error_bands(alpha, beta, n_params=n_params, n_grid=400)
            for k, val in spec.items():
                out[f"{tag}_{k}"] = val
            out[f"{tag}_eigs"] = eigs
            out[f"{tag}_weights"] = weights
            # 原始 Lanczos 三对角：存下来后任何误差带/L 判据的后处理都无需重跑 GPU 谱
            out[f"{tag}_alpha"] = alpha
            out[f"{tag}_beta"] = beta
            # ★ 逐曲线增量落盘：算完一条立刻存，可随时检查
            np.savez(args.out, **out)
            log(f"  [{tag}] eig[{eigs.min():.2e},{eigs.max():.2e}]  "
                f"cut={spec['cut']}  mid.max/N={spec['mid'].max()/n_params:.4f}  → 已落盘 {args.out}")
        dist.barrier()

    log(f"✅ 全部完成 → {args.out}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
