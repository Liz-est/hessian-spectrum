"""
单节点的历史脚本，用的是瞬时二阶矩而不是EMA

Lanczos 谱计算驱动（单 checkpoint，4 矩阵类型）

对一个 checkpoint 计算 4 条谱曲线（论文 Figure 2 约定）：
  GN Adam    (GN 矩阵，Adam 预条件)
  H Adam     (完整 Hessian，Adam 预条件)
  GN raw     (GN，恒等预条件)
  H raw      (H，恒等预条件)

输出格式与 spectrum_3x3.npz 一致：g/lo/hi/mid/L/R/cut 字段。

用法：
  python compute_spectrum.py <ckpt_path> --m 400 --n_tokens 65536 --out <output.npz>

示例：
  python compute_spectrum.py checkpoints_b64/ckpt_p100.pt --m 400 --out spectrum_b64_p100_m400.npz
"""
import os, sys, argparse, time
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.model.model import Transformer, TransformerConfig
from src.optim.opt import CompletePAdam
from src.spectrum.hvp import hessian_vector_product, gauss_newton_vector_product, build_adam_preconditioner
from src.spectrum.lanczos import lanczos_algorithm_1
from src.spectrum.gauss_radau import compute_spectrum_with_error_bands
from src.data.data_grain import make_hvp_batches

# 100BT parquet：谱分析 HVP 用与训练/原版谱脚本相同的 grain 采样。
from src import paths

DATA_DIR = str(paths.DATA_DIR)
N_PARAMS = 167_772_160


def load_checkpoint(path, device):
    """加载 checkpoint，返回 (model, opt_state_dict, config_dict)。"""
    ck = torch.load(path, map_location="cpu")
    cfg_dict = ck["config"]
    cfg = TransformerConfig(D=cfg_dict["D"], L=cfg_dict["L"], M=cfg_dict["M"],
                            H=cfg_dict["H"], K=cfg_dict["K"], V=cfg_dict["V"],
                            seq_len=cfg_dict["seq_len"])
    model = Transformer(cfg).to(device).float()
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["opt"], cfg_dict


def build_adam_precond_from_ckpt(opt_state, model, device):
    """从 checkpoint 的 opt state 重建 Adam 预条件器。

    严格对齐参考 preprocessing/sample_hessian_frob2.py:232-236：
        P = √(lr.pre · lr.post / (√ν̂ + ε))         —— 不含 base_lr
    其中 ν̂ = ν / (1 - Π β2)（bias correction）。

    ⚠ pre/post 从 model.lr_groups() 权威重建，不用 opt_state 里存的
    pre/post —— 早期 state_for_ckpt 有序列化 bug（post 误存成 pre），
    会把度量放大 ~D² 倍。lr_groups() 是纯 config 函数，不受影响。
    """
    nu_dict = opt_state["nu"]
    log_prod_b2 = opt_state["log_prod_b2"]
    eps = opt_state["eps"]
    # 权威 pre/post（含 CompleteP + BlockScales），与训练时一致
    lr_groups = {name: (pre, post) for name, _, pre, post in model.lr_groups()}

    # bias correction: ν̂ = ν / (1 - Π β2)
    denom = max(-np.expm1(log_prod_b2), 1e-16)
    nu_hat = {k: (v / denom).to(device) for k, v in nu_dict.items()}

    precond = {}
    for name, p in model.named_parameters():
        if name in nu_hat and name in lr_groups:
            pre, post = lr_groups[name]
            precond[name] = torch.sqrt((pre * post) / (torch.sqrt(nu_hat[name]) + eps))
        else:
            precond[name] = torch.ones_like(p)  # fallback
    return precond


def build_hvp_fn(model, batches, curvature, preconditioner, device):
    """构建批平均的 HVP 函数（curvature='gn'/'hessian', preconditioner=None/dict）。
    batches 为 make_hvp_batches 预采样的固定 minibatch 列表（grain 采样），
    HVP 在其上求和后除以 minibatch 数——每次 Lanczos matvec 用同一批数据。"""
    def hvp(v):
        acc = torch.zeros_like(v)
        for x, y in batches:
            if curvature == "gn":
                acc += gauss_newton_vector_product(model, x, y, v, preconditioner=preconditioner)
            else:
                acc += hessian_vector_product(model, x, y, v, preconditioner=preconditioner)
        return acc / len(batches)
    return hvp


def compute_one_curve(model, batches, m, curvature, preconditioner, device, seed):
    """计算一条谱曲线（Lanczos + Gauss-Radau）。"""
    hvp_fn = build_hvp_fn(model, batches, curvature, preconditioner, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Running Lanczos m={m} for {curvature} {'Adam' if preconditioner else 'raw'}...", flush=True)
    t0 = time.time()
    # Q(m×n_params) 太大（m=400 fp32 = 268 GB），存 CPU 内存，每步搬当前向量到 GPU 做 HVP
    eigs, weights, alpha, beta = lanczos_algorithm_1(
        hvp_fn, n_params, m, device, seed=seed, store_device="cpu")
    dt = time.time() - t0
    print(f"    Lanczos done in {dt:.1f}s, computing Gauss-Radau bands...", flush=True)
    spec = compute_spectrum_with_error_bands(alpha, beta, n_params=n_params, n_grid=400)
    return spec, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_path")
    ap.add_argument("--m", type=int, default=400)
    ap.add_argument("--n_tokens", type=int, default=65536, help="HVP 采样 token 数（batch×seq）")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {args.ckpt_path} on {device}...", flush=True)
    model, opt_state, cfg = load_checkpoint(args.ckpt_path, device)
    seq = cfg["seq_len"]
    print(f"Model: {cfg['D']}D {cfg['L']}L, n_params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    # grain 采样 HVP batch（与训练/原版谱脚本同一条 seed 流，shuffle=True+repeat）。
    # args.batch = 每个 HVP minibatch 的序列数（原版 run_dist 的 batch_size，默认小以控显存）。
    batches, n_seqs = make_hvp_batches(
        data_dir=DATA_DIR, seq_len=seq, vocab_size=cfg["V"],
        n_tokens=args.n_tokens, per=args.batch, device=device, seed=args.seed)
    print(f"HVP: {len(batches)} minibatch × {args.batch}×{seq} = {n_seqs*seq} tokens "
          f"(grain 采样，从流首取；未做原版的 iter.set_state / frob2 过滤)", flush=True)
    adam_precond = build_adam_precond_from_ckpt(opt_state, model, device)

    # 4 条曲线（论文 Figure 2 约定）
    curves = [
        ("gn",      adam_precond, "gn_adam"),
        ("hessian", adam_precond, "hessian_adam"),
        ("gn",      None,         "gn_sgd"),
        ("hessian", None,         "hessian_sgd"),
    ]

    results = {}
    for curvature, precond, label in curves:
        spec, dt = compute_one_curve(model, batches, args.m, curvature, precond, device, args.seed)
        for k in ("g", "lo", "hi", "mid", "L", "R", "cut", "x", "y", "dot_x"):
            results[f"{label}_{k}"] = spec[k]
        print(f"  [{label}] done: {dt:.1f}s, cut={spec['cut']}, mid range=[{spec['mid'].min():.1f}, {spec['mid'].max():.3e}]", flush=True)

    results["m"] = args.m
    results["n_tokens"] = n_seqs * seq
    np.savez(args.out, **results)
    print(f"\n✓ Saved {args.out}")


if __name__ == "__main__":
    main()
