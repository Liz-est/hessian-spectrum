"""
逐 unit 精确子块谱分析（QuadraticModel-rep 版「Hessian 路线一」编排）。

对给定 checkpoint，按 unit 粒度精确构造 GN/Fisher 块（blocks.UnitHessian）并特征分解，
再算层内 unit 间异质性（Symmetric-KL / JS 距离）。**不进 Lanczos**、无近似。

分析的 unit 工作项（--layer L 默认 5 = 第 6 层，1-indexed）：
  embedding      按 token（top-K 高频，见 block_token_freq）  Fisher
  lm_head        按 token(=class)（top-K 高频）              CE 真 Hessian 闭式
  L{L}.attn_q    按注意头（H=16）                            GN
  L{L}.attn_k    按注意头                                    GN
  L{L}.attn_v    按输出神经元（H·K=1024）                    Fisher
  L{L}.attn_o    按输出神经元（D=1024，=attn_head）          Fisher
  L{L}.mlp_fc    按输出神经元（M=4096，=mlp_up）             Fisher
  L{L}.mlp_proj  按输出神经元（D=1024，=mlp_head）           Fisher

复用 spectrum_ddp.load_checkpoint / make_local_batches（同 ckpt/EMA/数据口径）。
DDP：工作项按 rank strided 分片（大显存的 attn_q/k 尽量落不同 rank）。

产物 outputs/blocks_<STAMP>/（带 STAMP、绝不覆盖；summary_<name>.json 存在即跳过 resume）：
  eigs_<name>.npy         (n_units, d_block) 特征值
  hetero_<name>_skl.npy / _js.npy   (n_units, n_units) 距离矩阵
  summary_<name>.json     unit/n_units/d_block/λmax/trace/eff_rank/n_neg/hetero_mean...
  meta.json               全局：ckpt/layer/top_ids 来源/工作项清单

用法（单卡本地小样）:
  python analyze_blocks.py --ckpt checkpoints_b64/ckpt_p100.pt \
      --layer 5 --n_tokens 200000 --topk 64 --skip attn_q,attn_k \
      --token_freq outputs/token_freq_v8192.npz --out outputs/blocks_smoke
16 卡:
  torchrun --nnodes=2 --nproc_per_node=8 analyze_blocks.py --ckpt ... \
      --layer 5 --n_tokens 1000000 --topk 1024 \
      --token_freq outputs/token_freq_v8192.npz --out outputs/blocks_p100_0817
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks import UnitHessian                                  # noqa: E402
from hetero import common_log_edges, spectra_to_prob, pairwise_matrix, hetero_mean  # noqa: E402
from spectrum_ddp import load_checkpoint, make_local_batches    # noqa: E402


def summarize(ev):
    ev = np.asarray(ev, float)
    pos = ev[ev > 0]
    return {"n": int(ev.size), "lam_max": float(ev.max()), "lam_min": float(ev.min()),
            "trace": float(ev.sum()),
            "eff_rank": float(pos.sum() ** 2 / (pos ** 2).sum()) if pos.size else 0.0,
            "n_pos": int((ev > 0).sum()), "n_neg": int((ev < 0).sum())}


def build_worklist(layer, skip):
    """(display_name, kind, param_or_none)。kind ∈ token_emb/token_head/head/neuron。"""
    items = [
        ("embedding",           "token_emb",  None),
        ("lm_head",             "token_head", None),
        (f"L{layer:02d}.attn_q",   "head",   "attn_q"),
        (f"L{layer:02d}.attn_k",   "head",   "attn_k"),
        (f"L{layer:02d}.attn_v",   "neuron", "attn_v"),
        (f"L{layer:02d}.attn_o",   "neuron", "attn_head"),
        (f"L{layer:02d}.mlp_fc",   "neuron", "mlp_up"),
        (f"L{layer:02d}.mlp_proj", "neuron", "mlp_head"),
    ]
    sk = {s.strip() for s in skip.split(",") if s.strip()}
    return [it for it in items if it[0] not in sk and (it[2] or "") not in sk
            and it[1] not in sk]


def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist
        backend = "gloo" if not torch.cuda.is_available() else "nccl"
        dist.init_process_group(backend=backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        return rank, world, True
    return 0, 1, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, default=5, help="分析第几层（0-indexed；默认 5 = 第 6 层）")
    ap.add_argument("--token_freq", default="outputs/token_freq_v8192.npz",
                    help="block_token_freq.py 产物，取 top_ids 给 embedding/lm_head")
    ap.add_argument("--topk", type=int, default=1024, help="token 块取前 K 高频（≤ token_freq 里的 K）")
    ap.add_argument("--n_tokens", type=int, default=1_000_000)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--ema", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_bins", type=int, default=64, help="hetero 直方图 bin 数")
    ap.add_argument("--skip", default="", help="跳过的工作项/张量名，逗号分隔（如 attn_q,attn_k 省显存）")
    ap.add_argument("--cache_cpu", action="store_true", help="head 块的激活缓存放 CPU（省显存）")
    ap.add_argument("--out", required=True, help="产出目录（带 STAMP，绝不覆盖既有内容）")
    args = ap.parse_args()

    rank, world, is_ddp = setup_dist()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) \
        if torch.cuda.is_available() else torch.device("cpu")

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    os.makedirs(args.out, exist_ok=True)
    model, cfg, _p, _pr = load_checkpoint(args.ckpt, args.ema, device)

    # 每 rank 拿自己那份固定 batch，循环喂给 UnitHessian
    batches, nb_global = make_local_batches(
        cfg, args.n_tokens, world, rank, device, args.seed, per=args.per)
    log(f"world={world} device={device} nb_global={nb_global} "
        f"tokens/rank≈{len(batches)*args.per*cfg.seq_len:,}")
    _it = {"i": 0}
    def get_batch():
        b = batches[_it["i"] % len(batches)]
        _it["i"] += 1
        return b
    n_batches = len(batches)

    # top-K token ids
    tf = np.load(args.token_freq)
    top_ids = tf["top_ids"][:args.topk].astype(int)
    log(f"token 块用 top-{len(top_ids)}（来源 {args.token_freq}）")

    uh = UnitHessian(model, get_batch, n_batches=n_batches, device=device,
                     cache_device="cpu" if args.cache_cpu else device)

    work = build_worklist(args.layer, args.skip)
    my_work = work[rank::world] if is_ddp else work
    if rank == 0:
        json.dump({"ckpt": args.ckpt, "layer": args.layer, "topk": int(len(top_ids)),
                   "token_freq": args.token_freq, "n_tokens": args.n_tokens,
                   "world": world, "items": [w[0] for w in work]},
                  open(os.path.join(args.out, "meta.json"), "w"), indent=2)

    for disp, kind, param in my_work:
        sfile = os.path.join(args.out, f"summary_{disp}.json")
        if os.path.exists(sfile):
            log(f"[skip] {disp} 已完成")
            continue
        print(f"[rank{rank}] 计算 {disp} ({kind})...", flush=True)

        if kind == "token_emb":
            eigs, meta = uh.embedding_token_blocks(token_ids=top_ids)
        elif kind == "token_head":
            eigs, meta = uh.lm_head_token_blocks(token_ids=top_ids)
        elif kind == "head":
            eigs, meta = uh.head_blocks(param, args.layer)
        else:  # neuron
            eigs, meta = uh.neuron_blocks(param, args.layer)

        np.save(os.path.join(args.out, f"eigs_{disp}.npy"), eigs)

        # 层内 unit 间异质性（unit 数太多时 JS 的 n² 会大，这里 unit 数 ≤4096，可接受）
        edges = common_log_edges(eigs, args.num_bins)
        P = spectra_to_prob(eigs, edges)
        D_skl = pairwise_matrix(P, "skl", device=device)
        D_js = pairwise_matrix(P, "js", device=device)
        np.save(os.path.join(args.out, f"hetero_{disp}_skl.npy"), D_skl)
        np.save(os.path.join(args.out, f"hetero_{disp}_js.npy"), D_js)

        info = {"disp": disp, "kind": kind, "param": param,
                "unit": meta.get("unit"), "n_units": int(eigs.shape[0]),
                "d_block": int(eigs.shape[1]), "n_tok": int(meta.get("n_tok", 0)),
                "skl_mean": hetero_mean(D_skl), "js_mean": hetero_mean(D_js),
                "pooled": summarize(eigs)}
        if "counts" in meta:
            info["token_ids"] = [int(t) for t in top_ids]
        json.dump(info, open(sfile, "w"), indent=2)
        print(f"[rank{rank}] ✓ {disp}: n_units={eigs.shape[0]} d={eigs.shape[1]} "
              f"λmax={info['pooled']['lam_max']:.3e} skl={info['skl_mean']:.3f}", flush=True)

    if is_ddp:
        import torch.distributed as dist
        dist.barrier()
        if rank == 0:
            log("✅ 全部 rank 完成")
        dist.destroy_process_group()
    else:
        log("✅ 完成")


if __name__ == "__main__":
    main()
