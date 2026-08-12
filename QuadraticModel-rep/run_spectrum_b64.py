"""
③f: B=64 checkpoint 的 Figure-2 谱计算（4 曲线 × 每 checkpoint）。

对齐论文预处理 QuadraticModel/preprocessing/sample_hessian_frob2.py：
  - 用 EMA'd 参数 + EMA'd ν（默认 ema=0.04）
  - Adam 预条件器 P = √(lr.pre·lr.post/(√ν̂ + eps))，对称作用 P·H·P / P·G·P
  - ν̂ = bias_correct(EMA'd ν, log_prod_b2)
  - "adam" 曲线用该预条件器；"sgd" 曲线 = 恒等（裸 H/G）
  - GN 用 JVP+centered+VJP；Hessian 用 double-backward
  - Lanczos Algorithm 1（reorth-before-normalize），m 深度可调
  - Gauss-Radau 误差带（Golub-Meurant 端点锚定）

每 checkpoint 输出 4 条曲线：gn_adam, hessian_adam, gn_sgd, hessian_sgd。
结果字段与缓存 spectrum_3x3.npz 对齐（g/lo/hi/mid/x/y/L/R/cut）。

用法（GPU worker）：
  python run_spectrum_b64.py --ckpt checkpoints_b64/ckpt_p100.pt --m 400 \
      --hvp_tokens 65536 --ema 0.04 --out test_outputs/spectrum_b64_p100.npz
"""
import os, sys, time, math, argparse, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, TransformerConfig, param_group_slices
from hvp import hessian_vector_product, gauss_newton_vector_product
from lanczos import lanczos_algorithm_1
from gauss_radau import compute_spectrum_with_error_bands

DATA_DIR = "/data/250010020/hessian-spectrum/data/fineweb_edu_bpe8192"


def load_checkpoint(path, ema_key, device):
    ck = torch.load(path, map_location="cpu")
    c = ck["config"]
    cfg = TransformerConfig(D=c["D"], L=c["L"], M=c["M"], H=c["H"], K=c["K"],
                            V=c["V"], seq_len=c["seq_len"])
    model = Transformer(cfg)

    # 用 EMA'd 参数（若存在），否则 last-iterate
    ema = ck.get("ema")
    if ema is not None and str(ema_key) in ema:
        sd = model.state_dict()
        for name, v in ema[str(ema_key)].items():
            sd[name] = v
        model.load_state_dict(sd)
        print(f"  使用 EMA 参数 ema={ema_key}")
    else:
        model.load_state_dict(ck["model"])
        print("  使用 last-iterate 参数")
    model.eval().to(device)

    # 重建 Adam 预条件器：ν̂ = EMA'd ν / (1 - Πβ2)
    opt = ck["opt"]
    log_prod_b2 = ck.get("log_prod_b2", opt.get("log_prod_b2", 0.0))
    denom_b2 = max(-math.expm1(log_prod_b2), 1e-16)
    nu_ema = ck.get("nu_ema")
    if nu_ema is not None and str(ema_key) in nu_ema:
        nu_src = nu_ema[str(ema_key)]
        print(f"  使用 EMA'd ν ema={ema_key}")
    else:
        nu_src = opt["nu"]
        print("  使用 last ν")
    eps = opt["eps"]; pre = opt["pre"]; post = opt["post"]
    precond = {}
    for name, _ in model.named_parameters():
        nu_hat = nu_src[name].to(device).float() / denom_b2
        precond[name] = torch.sqrt(pre[name] * post[name] / (torch.sqrt(nu_hat) + eps))
    return model, cfg, precond


def make_batches(cfg, n_tokens, device, seed):
    """从 val/train 流采样固定 batch 集，HVP 上平均。"""
    data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
    seq = cfg.seq_len
    n_seqs = max(1, n_tokens // seq)
    g = torch.Generator().manual_seed(seed)
    batches = []
    # 每 batch 8 条序列，控显存
    per = 8
    for s in range(0, n_seqs, per):
        bs = min(per, n_seqs - s)
        ix = torch.randint(len(data) - seq - 1, (bs,), generator=g)
        x = torch.stack([torch.from_numpy(data[i:i+seq].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i+1:i+1+seq].astype(np.int64)) for i in ix]).to(device)
        batches.append((x, y))
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--m", type=int, default=400)
    ap.add_argument("--hvp_tokens", type=int, default=65536)  # 1 minibatch B=64×1024
    ap.add_argument("--ema", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备 {device}  m={args.m}  hvp_tokens={args.hvp_tokens}  ema={args.ema}")
    model, cfg, precond = load_checkpoint(args.ckpt, args.ema, device)
    n_params = model.n_params()
    batches = make_batches(cfg, args.hvp_tokens, device, args.seed)
    print(f"  HVP 平均 batch 数={len(batches)}  n_params={n_params:,}")

    def make_hvp(kind, use_precond):
        p = precond if use_precond else None
        def fn(v):
            acc = torch.zeros(n_params, device=device)
            for x, y in batches:
                if kind == "gn":
                    acc = acc + gauss_newton_vector_product(model, x, y, v, preconditioner=p)
                else:
                    acc = acc + hessian_vector_product(model, x, y, v, preconditioner=p)
            return acc / len(batches)
        return fn

    CURVES = [("gn", True, "gn_adam"), ("hessian", True, "hessian_adam"),
              ("gn", False, "gn_sgd"), ("hessian", False, "hessian_sgd")]
    out = {}
    for kind, use_p, tag in CURVES:
        t0 = time.time()
        hvp = make_hvp(kind, use_p)
        eigs, weights, alpha, beta = lanczos_algorithm_1(
            hvp_fn=hvp, n_params=n_params, m=args.m, device=device, seed=args.seed)
        spec = compute_spectrum_with_error_bands(alpha, beta, n_params=n_params, n_grid=400)
        for k, val in spec.items():
            out[f"{tag}_{k}"] = val
        out[f"{tag}_eigs"] = eigs
        out[f"{tag}_weights"] = weights
        print(f"  {tag}: {time.time()-t0:.0f}s  eig[{eigs.min():.2e},{eigs.max():.2e}]  "
              f"cut={spec['cut']}  mid.max/N={spec['mid'].max()/n_params:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, m=args.m, n_params=n_params, hvp_tokens=args.hvp_tokens,
             ema=args.ema, **out)
    print(f"✅ 保存 {args.out}")


if __name__ == "__main__":
    main()
