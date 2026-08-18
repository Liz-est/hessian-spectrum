"""
统计 V=8192 词表下最高频的 token（供 embedding / lm_head 的按-token 分块用）。

背景：QuadraticModel-rep 用自训 bpe_8192 tokenizer（V=8192），toy_models 那份
`token_counts.npy` 是 50304 词表、不适用。这里从**训练同一条 grain 流**
（spectrum_ddp.make_local_batches 的口径）采样 N 个 token，bincount 累加，取前 K。

产物 `<out>`（默认 outputs/token_freq_v8192.npz），含：
  counts   (V,) int64   —— 每个 token id 的经验计数
  top_ids  (K,) int64   —— 按计数降序的前 K 个 token id
  n_tokens int          —— 实际统计的 token 总数
  K, V     int
**已存在则跳过不覆盖**（贯穿本项目的硬约束）。embedding 与 lm_head 共用同一份 top_ids。

用法:
  python block_token_freq.py --ckpt checkpoints_b64/ckpt_p100.pt \
      --n_tokens 2000000 --topk 1024 --out outputs/token_freq_v8192.npz
（只需 ckpt 里的 config 拿 V/seq_len，不做前向、不上多卡。）
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src import paths  # noqa: E402
from src.spectrum.spectrum_ddp import load_checkpoint, make_local_batches   # noqa: E402


def count_tokens(cfg, n_tokens, device, seed, per):
    """从 grain 流采样，累加 input-token(x) 的 bincount。返回 (counts, n_seen)。"""
    # world=1 / rank=0：单流顺序取全部 minibatch（与谱分析同一批确定性样本）
    batches, _nb = make_local_batches(cfg, n_tokens, 1, 0, device, seed, per=per)
    counts = torch.zeros(cfg.V, dtype=torch.int64, device=device)
    n_seen = 0
    for x, _y in batches:
        flat = x.reshape(-1)
        counts += torch.bincount(flat, minlength=cfg.V)
        n_seen += flat.numel()
    return counts.cpu().numpy(), n_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="只用它的 config 取 V/seq_len")
    ap.add_argument("--n_tokens", type=int, default=2_000_000)
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ema", type=float, default=0.04)
    ap.add_argument("--out", default=str(paths.OUT_DIR / "token_freq_v8192.npz"))
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"[skip] 已存在，不覆盖: {args.out}")
        d = np.load(args.out)
        print(f"  counts{d['counts'].shape}  top_ids{d['top_ids'].shape}  "
              f"n_tokens={int(d['n_tokens'])}")
        return

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model, cfg, _p, _pr = load_checkpoint(args.ckpt, args.ema, device)
    del model  # 只要 cfg

    counts, n_seen = count_tokens(cfg, args.n_tokens, device, args.seed, args.per)
    top_ids = np.argsort(-counts)[:args.topk].astype(np.int64)

    nz = int((counts > 0).sum())
    print(f"统计 {n_seen:,} token；{nz}/{cfg.V} 个 token 至少出现一次", flush=True)
    print(f"top-{min(8, args.topk)} id: {top_ids[:8].tolist()}  "
          f"counts: {counts[top_ids[:8]].tolist()}", flush=True)
    print(f"第 {args.topk} 名 count={int(counts[top_ids[-1]])}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, counts=counts, top_ids=top_ids,
             n_tokens=n_seen, K=args.topk, V=cfg.V)
    print(f"→ {args.out}", flush=True)


if __name__ == "__main__":
    main()
