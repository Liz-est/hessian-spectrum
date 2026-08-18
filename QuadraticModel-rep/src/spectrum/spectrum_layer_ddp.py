"""
块 Hessian / 块 GN 谱（分布式，lockstep）。**不改动 spectrum_ddp.py 的全量逻辑**，
复用它的 load_checkpoint / make_local_batches / 4 曲线定义与落盘约定。

数学：block b 的谱 = H_bb = P_bᵀ H P_b 的谱，等于「只有块 b 可训练」子问题的 Hessian
谱，不是近似。它**不能**从全量谱反推：
    (H^k)_bb = P_bᵀ H  I  H ⋯ P_b        ← 全量 Lanczos 的矩量能给的
    H_bb^k   = P_bᵀ H Π_b H ⋯ Π_b H P_b  ← 块谱需要的（Π_b 是块投影）
从 k=2 起就分叉：(H²)_bb = H_bb² + Σ_{a≠b} H_ba H_ab，多出的交叉项恒 PSD。
见 verify_layers.py 的定量核对。

用法（16 卡，全量 + 14 个层块，GN adam 一条曲线）：
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$RANK \
      --master_addr=$MASTER_ADDR --master_port=29500 \
      spectrum_layer_ddp.py --ckpt checkpoints_b64/ckpt_p100.pt \
      --layers layers --m 400 --curves gn_adam \
      --out outputs/blocks_p100_gn_adam.npz

本地 sanity（2 rank CPU）：
  torchrun --standalone --nproc_per_node=2 spectrum_layer_ddp.py --ckpt ... \
      --backend gloo --cpu --layers embd,head --m 20 --n_tokens 8192
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.model.layers import build_layers, check_partition                      # noqa: E402
from src.spectrum.gauss_radau import compute_spectrum_with_error_bands             # noqa: E402
from src.spectrum.hvp_layers import make_layer_matvec                              # noqa: E402
from src.spectrum.lanczos_layers import lockstep_lanczos                           # noqa: E402
# 复用全量脚本的 checkpoint / 数据逻辑，保证两条路径口径完全一致
from src.spectrum.spectrum_ddp import load_checkpoint, make_local_batches          # noqa: E402

# 与 spectrum_ddp.CURVES 同名同义：(kind, precond_sel, tag)
CURVES = [("gn", "adam", "gn_adam"), ("hessian", "adam", "hessian_adam"),
          ("gn", "raw", "gn_raw"), ("hessian", "raw", "hessian_raw")]


def is_master():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(*a):
    if is_master():
        print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    # --layers 是主参数；--groups 保留为兼容别名（旧脚本/缓存仍能用）。
    ap.add_argument("--layers", "--groups", dest="layers", default="layers",
                    help="层/张量分组，见 layers.build_layers（full/layers/layer-parts/"
                         "layer-tensors/tensors/embd/layerNN[.attn|.mlp|.headHH]，逗号分隔）")
    ap.add_argument("--m", type=int, default=400, help="每块 Lanczos 步数")
    ap.add_argument("--m_full", type=int, default=1200,
                    help="name=full 的块用这个 m（全量维度大，需要更多步）")
    ap.add_argument("--n_tokens", type=int, default=1_000_000)
    ap.add_argument("--ema", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--max_lockstep", type=int, default=0,
                    help="每波最多同时跑几块（0=全部）。Q 显存不够时分波，"
                         "避免 Q 回退 CPU 让重正交 O(m²n) 变瓶颈")
    ap.add_argument("--curves", default="gn_adam",
                    help=f"逗号分隔，可选 {[t for *_, t in CURVES]}，或 all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    dist.init_process_group(backend=args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.cpu:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    store_device = torch.device("cpu")

    log(f"world={world} backend={args.backend} device={device}")
    log(f"ckpt={args.ckpt}  m={args.m} (full:{args.m_full})  layers={args.layers}")

    model, cfg, precond, precond_raw = load_checkpoint(args.ckpt, args.ema, device)
    layers = build_layers(model, args.layers)
    log(f"  n_params={model.n_params():,}   {len(layers)} 块:")
    for b in layers:
        log(f"    {b.name:18s} n_b={b.numel:>11,}  ({b.numel/model.n_params()*100:5.2f}%)"
            f"  params={b.params}")
    if not any(b.name == "full" for b in layers):
        cov, tot, ov = check_partition(model, layers)
        log(f"  覆盖 {cov:,}/{tot:,} ({cov/tot*100:.1f}%)  重叠 {ov:,}")

    local_batches, nb_global = make_local_batches(
        cfg, args.n_tokens, world, rank, device, args.seed, per=args.per)
    log(f"  HVP: nb_global={nb_global} × per={args.per} × {cfg.seq_len} = "
        f"{nb_global*args.per*cfg.seq_len:,} tokens")

    curves = CURVES if args.curves == "all" else \
        [c for c in CURVES if c[2] in {t.strip() for t in args.curves.split(",")}]
    if not curves:
        raise SystemExit(f"--curves 未匹配任何曲线；可选 {[t for *_, t in CURVES]}")
    log(f"  曲线: {[t for *_, t in curves]}")

    def m_of(b):
        return args.m_full if b.name == "full" else args.m

    # HVP 双反向峰值 ≈ per×3.4GB + 4GB 底（与 spectrum_ddp 同式）
    reserve = int((args.per * 3.4 + 4) * 1024**3)

    waves = ([layers] if args.max_lockstep <= 0 else
             [layers[i:i + args.max_lockstep]
              for i in range(0, len(layers), args.max_lockstep)])

    out = {"m": args.m, "m_full": args.m_full, "n_params": model.n_params(),
           "n_tokens": nb_global * args.per * cfg.seq_len, "ema": args.ema,
           "groups": args.layers, "raw_precond": "completep",
           "block_names": np.array([b.name for b in layers]),
           "block_numel": np.array([b.numel for b in layers])}
    if is_master():
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for kind, sel, tag in curves:
        p = precond if sel == "adam" else precond_raw
        matvec = make_layer_matvec(model, local_batches, nb_global, kind, p, dist=dist)
        for wi, wave in enumerate(waves):
            dist.barrier()
            t0 = time.time()
            log(f"\n=== 曲线 {tag}  第 {wi+1}/{len(waves)} 波 "
                f"({len(wave)} 块: {', '.join(b.name for b in wave)}) ===")
            tri = lockstep_lanczos(matvec, wave, m_of, world, rank, device,
                                   store_device, args.seed, reserve, log=log)
            dist.barrier()
            log(f"  本波耗时 {time.time()-t0:.0f}s")

            if is_master():
                for b in wave:
                    alpha, beta = tri[b.name]
                    # ⚠ n_params 必须换成 n_b：index = N × 质量，权重恒 Σ=1，
                    # 用错 N 纵轴会整体偏 N/n_b 倍
                    spec = compute_spectrum_with_error_bands(
                        alpha, beta, n_params=b.numel, n_grid=400)
                    pre = f"{tag}_{b.name}"
                    for k, val in spec.items():
                        out[f"{pre}_{k}"] = val
                    out[f"{pre}_alpha"] = alpha
                    out[f"{pre}_beta"] = beta
                    out[f"{pre}_numel"] = b.numel
                    ev, _ = np.linalg.eigh(
                        np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1))
                    out[f"{pre}_eigs"] = ev
                    log(f"  [{pre}] λ∈[{ev.min():.3e},{ev.max():.3e}]  "
                        f"cut={spec['cut']}  L={spec['L']}")
                np.savez(args.out, **out)     # 逐波增量落盘
                log(f"  → 已落盘 {args.out}")
            dist.barrier()

    log(f"\n✅ 全部完成 → {args.out}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
