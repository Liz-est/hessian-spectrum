"""
二号猜疑验证：frob2 过滤对 raw Hessian 谱的 A/B 实验（16 卡 = 2×8 H100）。

流程（单变量 = 是否剔除 Adam-precond GN-frob2 的 top-1%）：
  1. 用与 baseline 完全相同的 grain 口径建**单条序列**池（per=1，seed 固定），
     所有 rank 建出逐条一致的池。
  2. 分布式给每条序列打 Adam-precond GN-frob2 分（frob2_score，探针 Rademacher，
     全局下标定种子 → 与 rank 划分无关、可复现）；all_reduce 汇总到完整分数向量。
  3. threshold = quantile(scores, 0.99)；keep99 = scores < threshold（保留 99%），
     keepall = 全部（baseline，走同一代码路径消除混淆）。
  4. 对 keepall / keep99 两个子集各把序列按 per 组成 minibatch、分片到各 rank，
     跑 **hessian_raw**（CompleteP 预条件的 Hessian）Lanczos 谱。
  5. rank0 落盘：两条谱 + 全部分数 + threshold + 保留下标。

判读：keep99 尾部明显下压向论文 → 二号成立；两条重合 → 二号在真实数据也证伪。

复用 spectrum_ddp 的：load_checkpoint / shard_bounds / make_dist_hvp / lanczos_sharded；
复用 data_grain.make_hvp_batches 建序列池；复用 frob2_score 打分。

启动（2 节点 × 8 卡）：
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$RANK \
      --master_addr=$MASTER_ADDR --master_port=29500 \
      spectrum_frob2_ddp.py --ckpt checkpoints_b64/ckpt_p100.pt \
      --m 1200 --n_tokens 1000000 --per 2 --num_probes 10 \
      --out outputs/frob2_ab_p100.npz

本地 sanity（2 rank CPU）：
  torchrun --standalone --nproc_per_node=2 spectrum_frob2_ddp.py --ckpt ... \
      --backend gloo --cpu --m 6 --n_tokens 12288 --per 2 --num_probes 2 --out /tmp/ab.npz
"""
import os, sys, time, math, argparse
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.spectrum.spectrum_ddp import (
    setup_dist, is_master, log, log_all, shard_bounds,
    load_checkpoint, make_dist_hvp, lanczos_sharded, DATA_DIR,
)
from src.data.data_grain import make_hvp_batches
from src.spectrum.frob2_score import score_sequences
from src.spectrum.gauss_radau import compute_spectrum_with_error_bands


# --------------------------------------------------------------------------
# 建序列池：**精确复刻 baseline（full_hessian_raw.npz）的那 992 条序列**，
# 使 keep99 谱能与已有 baseline 直接对比（不重跑不过滤的谱）。
#
# baseline 由 spectrum_ddp.make_local_batches 建：
#   n_minibatch = max(ref_world, n_tokens//seq//ref_per)
#   nb_global   = ceil(n_minibatch/ref_world)*ref_world      （向上取到 ref_world 倍）
#   池 = make_hvp_batches(n_tokens=nb_global*ref_per*seq, per=ref_per, seed) 的 nb_global 组
# 这里用**同一公式、同一 grain 流、同一 seed**建出同样的 nb_global 组，再把每组
# ref_per 条展平成单序列列表 → 992 条，与 baseline 逐条一致（grain 窗口与 batch 大小
# 无关，故 per=ref_per 展平 == baseline 池）。ref_world/ref_per 固定为 baseline 的
# 16/2，与本次实际 world 无关，保证任何 world 下池都相同。
# --------------------------------------------------------------------------
def build_sequence_pool(cfg, n_tokens, device, seed, ref_world=16, ref_per=2):
    seq = cfg.seq_len
    n_minibatch = max(ref_world, n_tokens // seq // ref_per)
    nb_global = ((n_minibatch + ref_world - 1) // ref_world) * ref_world
    pool_tokens = nb_global * ref_per * seq
    mbs, _ = make_hvp_batches(
        data_dir=DATA_DIR, seq_len=seq, vocab_size=cfg.V,
        n_tokens=pool_tokens, per=ref_per, device=device, seed=seed)
    seqs = []
    for xs, ys in mbs:                         # nb_global 组，每组 (ref_per, seq)
        for r in range(xs.shape[0]):
            seqs.append((xs[r:r + 1], ys[r:r + 1]))  # 展平为单序列 (1, seq)
    return seqs


# --------------------------------------------------------------------------
# 分布式打分：rank r 负责 g % world == r 的序列，all_reduce 汇总
# --------------------------------------------------------------------------
def distributed_scores(model, seqs, precond, num_probes, probe_seed, device, n_params, world, rank):
    n = len(seqs)
    my_idxs = [g for g in range(n) if g % world == rank]
    local = score_sequences(
        model, seqs, precond, num_probes, probe_seed, device, n_params,
        idxs=my_idxs, log_every=25, logfn=log_all)
    scores = torch.zeros(n, dtype=torch.float64, device=device)
    for g, s in local.items():
        scores[g] = s
    dist.all_reduce(scores, op=dist.ReduceOp.SUM)
    return scores.cpu().numpy()  # (n,) 完整分数，各 rank 一致


# --------------------------------------------------------------------------
# 由保留序列子集组 minibatch 并分片到各 rank（block 分配，尾部不足 per 的丢弃）
#   与 spectrum_ddp.make_local_batches 的归一化约定一致：
#   每 minibatch HVP 是逐 token 均值，all_reduce 求和后 /nb_global = grand mean。
#   要求所有 minibatch 同 per（同 token 数），故丢尾部不足一组的残余。
# --------------------------------------------------------------------------
def make_local_batches_from_kept(seqs, kept_idxs, per, world, rank):
    kept = sorted(kept_idxs)
    n_full = len(kept) // per
    dropped = len(kept) - n_full * per
    minibatches = []
    for j in range(n_full):
        group = kept[j * per:(j + 1) * per]
        xs = torch.cat([seqs[g][0] for g in group], dim=0)  # (per, seq_len)
        ys = torch.cat([seqs[g][1] for g in group], dim=0)
        minibatches.append((xs, ys))
    nb_global = n_full
    # block 分配：shard_bounds 把 [0, nb_global) 尽量均分给各 rank
    bnds = shard_bounds(nb_global, world) if nb_global >= world else None
    if bnds is None:
        # minibatch 数 < world：让前 nb_global 个 rank 各拿 1 个，其余空
        local = [minibatches[rank]] if rank < nb_global else []
    else:
        s, e = bnds[rank]
        local = minibatches[s:e]
    return local, nb_global, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--m", type=int, default=1200)
    ap.add_argument("--n_tokens", type=int, default=1_000_000)
    ap.add_argument("--ema", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42, help="序列池 grain 种子（须与 baseline 一致）")
    ap.add_argument("--per", type=int, default=2, help="每 HVP minibatch 序列数")
    ap.add_argument("--num_probes", type=int, default=10, help="frob2 Hutchinson 探针数")
    ap.add_argument("--probe_seed", type=int, default=0)
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    rank, world = setup_dist(args.backend)
    if args.cpu:
        device = torch.device("cpu")
    else:
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    store_device = torch.device("cpu")

    log(f"world={world} backend={args.backend} device={device}")
    log(f"m={args.m} n_tokens={args.n_tokens} per={args.per} num_probes={args.num_probes} "
        f"quantile={args.quantile} ckpt={args.ckpt}")

    model, cfg, precond, precond_raw = load_checkpoint(args.ckpt, args.ema, device)
    n_params = model.n_params()
    bounds = shard_bounds(n_params, world)
    log(f"  n_params={n_params:,}")

    # ---- 1) 建序列池 ----
    seqs = build_sequence_pool(cfg, args.n_tokens, device, args.seed)
    n_seqs = len(seqs)
    log(f"  序列池：{n_seqs} 条 × {cfg.seq_len} = {n_seqs*cfg.seq_len:,} tokens")

    # ---- 2) Adam-precond GN frob2 打分（论文的唯一过滤器）----
    dist.barrier()
    t0 = time.time()
    log("\n=== frob2 打分（Adam 预条件 GN）===")
    scores = distributed_scores(model, seqs, precond, args.num_probes,
                                args.probe_seed, device, n_params, world, rank)
    log(f"  打分完成 {time.time()-t0:.0f}s  "
        f"min={scores.min():.3e} med={np.median(scores):.3e} max={scores.max():.3e}")

    # ---- 3) q99 切 mask ----
    threshold = float(np.quantile(scores, args.quantile))
    keep99 = [g for g in range(n_seqs) if scores[g] < threshold]
    n_removed = n_seqs - len(keep99)
    log(f"  threshold(q{args.quantile})={threshold:.4e}  "
        f"keep99={len(keep99)}  removed={n_removed}")

    out = {"m": args.m, "n_params": n_params, "ema": args.ema, "seed": args.seed,
           "per": args.per, "num_probes": args.num_probes, "quantile": args.quantile,
           "n_seqs": n_seqs, "seq_len": cfg.seq_len,
           "frob2_scores": scores, "threshold": threshold,
           "keep99_idx": np.array(keep99, dtype=np.int64),
           "raw_precond": "completep"}
    if is_master():
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    hvp_reserve = int((args.per * 3.4 + 4) * 1024**3)

    # ---- 4) 只跑 keep99 一条 hessian_raw 谱（baseline 不重跑，用已有 full_hessian_raw.npz）----
    VARIANTS = [("keep99", keep99)]
    for name, kept in VARIANTS:
        dist.barrier()
        t0 = time.time()
        local_batches, nb_global, dropped = make_local_batches_from_kept(
            seqs, kept, args.per, world, rank)
        tokens_actual = nb_global * args.per * cfg.seq_len
        log(f"\n=== 谱 hessian_raw[{name}] ===  "
            f"minibatch={nb_global}×{args.per}  tokens={tokens_actual:,}  丢尾部 {dropped} 条")
        hvp_fn = make_dist_hvp(model, local_batches, nb_global, "hessian", precond_raw, device)
        eigs, weights, alpha, beta = lanczos_sharded(
            hvp_fn, n_params, args.m, world, rank, bounds, device,
            store_device, args.seed, hvp_reserve_bytes=hvp_reserve)
        dist.barrier()
        log_all(f"谱 {name} 完成 {time.time()-t0:.0f}s")
        if is_master():
            spec = compute_spectrum_with_error_bands(alpha, beta, n_params=n_params, n_grid=400)
            for k, val in spec.items():
                out[f"{name}_{k}"] = val
            out[f"{name}_eigs"] = eigs
            out[f"{name}_weights"] = weights
            out[f"{name}_alpha"] = alpha
            out[f"{name}_beta"] = beta
            out[f"{name}_n_tokens"] = tokens_actual
            np.savez(args.out, **out)
            log(f"  [{name}] eig[{eigs.min():.2e},{eigs.max():.2e}]  "
                f"cut={spec['cut']}  mid.max/N={spec['mid'].max()/n_params:.4f}  → 落盘 {args.out}")
        dist.barrier()

    log("\n全部完成。")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
