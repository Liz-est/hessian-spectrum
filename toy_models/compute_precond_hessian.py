#!/usr/bin/env python
"""Preconditioned-Hessian block heterogeneity for the 0-layer toy model.

The target is NOT the raw Hessian H but the PRECONDITIONED Hessian P^{-1}H,
where P is the optimizer's preconditioner. This is what actually differs across
SGD / Adam / Muon (the raw H is optimizer-independent), and it is where the
"does the optimizer make a block's spectrum more uniform?" question lives.

LAYER-AGNOSTIC by design (no frozen-embedding hardcoding): for the 0-layer
model both trainable weight matrices admit an exact block Hessian that factors
as a SHARED (d,d) matrix M times a per-block scalar scale_k:

    lm_head  (blocks = output classes k):
        H_k = (1/N) sum_t x_t x_t^T           (x_t = lm_head input activation)
        M = that Gram,  scale_k = 1           (class-independent; exact for
                                               mse/mse_rep, GN for ce is skipped)
    embedding (blocks = tokens v, 0-layer GN, exact whether or not embedding
               is frozen -- z = W e_v is LINEAR in e_v so H_v = GN block):
        H_v = (N_v/N) * W^T W                  (W = lm_head.weight, current)
        M = W^T W,     scale_v = N_v/N         (token frequency)

Preconditioners (derived from config/build.py's actual implementations):
    SGD:  P = I.
    Adam: per-parameter diagonal. For block k (row k of the analyzed weight),
          P_k = diag(sqrt(v_k) + eps), v_k = EMA_{beta2}[g_k^2] replayed from
          init. Preconditioned block = scale_k * D_k M D_k, D_k = P_k^{-1/2}.
    Muon: P = (G^T G)^{1/2} realized NS5-exactly as NS5(G)^T @ G (accounts for
          the Newton-Schulz approximation the optimizer really uses). Shared
          across blocks -> preconditioned block = scale_k * (P^{-1/2} M P^{-1/2}).

Per checkpoint we emit two axes:
    X (between-block): std of log10(per-block mean eigenvalue).
    Y (within-block):  mean over blocks of spectral entropy
                       H = -sum(l~ log l~)/log d, l~ = lambda/sum(lambda)
                       (1 = perfectly uniform spectrum).

Because every block is scale_k * (a shared preconditioned matrix) for SGD/Muon,
and scale_k * D_k M D_k for Adam, spectral entropy is scale-invariant: for
SGD/Muon it is identical across blocks (one eigh); for Adam it varies with D_k
(one eigh per subsampled block). The between-block mean uses ALL blocks cheaply
(trace, no eigh).

Usage (run on a GPU box / SCO H100):
    python compute_precond_hessian.py <run_dir> --layer lm_head --optim adam
    python compute_precond_hessian.py <run_dir> --layer embedding --optim muon
"""
import argparse
import os
import json

import numpy as np
import torch

from vanilla_model import ToyVanilla, ToyVanillaConfig

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
EPS = 1e-8
TAG_ORDER = ["init", "p10", "p25", "p40", "p50", "p60", "p75", "p85", "p100"]

# which weight tensor each analyzable layer maps to
LAYER_PARAM = {"lm_head": "lm_head.weight", "embedding": "tok_emb.weight"}


# ---------------------------------------------------------------------------
# Newton-Schulz (verbatim from config/build.py) -- Muon's real orthogonalizer
# ---------------------------------------------------------------------------
def _zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


# ---------------------------------------------------------------------------
# checkpoint / data / gradient helpers
# ---------------------------------------------------------------------------
def load_model_and_ckpt(run_dir, tag, device="cpu"):
    ckpt_path = os.path.join(run_dir, f"ckpt_{tag}.pt")
    if not os.path.exists(ckpt_path):
        return None, None
    ck = torch.load(ckpt_path, map_location=device)
    cfg_dict = dict(ck["config"])
    exp_dict = ck.get("experiment", {})
    valid = ToyVanillaConfig.__dataclass_fields__.keys()
    cfg_kwargs = {k: v for k, v in cfg_dict.items() if k in valid}
    cfg_kwargs["device"] = device
    model = ToyVanilla(ToyVanillaConfig(**cfg_kwargs)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, exp_dict


def get_dataloader_one_batch(exp_dict, block_size, device="cpu"):
    """Deterministic full-batch loader (first N tokens) matching the run's data."""
    data_cfg = exp_dict["data"]
    dataset = data_cfg["dataset"]
    batch_size = data_cfg["batch_size"]
    data_format = data_cfg.get("format", "dual_stream")
    if data_format != "dual_stream":
        raise NotImplementedError(f"format={data_format} not supported")
    data_dir = os.path.join(REPO_ROOT, "data", dataset)
    xd = np.memmap(os.path.join(data_dir, "train_x.bin"), dtype=np.uint16, mode="r")
    yd = np.memmap(os.path.join(data_dir, "train_y.bin"), dtype=np.uint16, mode="r")

    def get_batch():
        need = min(batch_size * block_size, (len(xd) // block_size) * block_size)
        x = torch.from_numpy(xd[:need].astype(np.int64)).view(-1, block_size)
        y = torch.from_numpy(yd[:need].astype(np.int64)).view(-1, block_size)
        return x.to(device), y.to(device)

    return get_batch


def _get_param(model, param_path):
    m = model
    for a in param_path.split("."):
        m = getattr(m, a)
    return m


def compute_gradient_fullbatch(model, get_batch, param_path, chunk_seqs=None):
    """Full-batch gradient of loss w.r.t. the analyzed weight tensor. Returns
    (n_rows, d) on CPU fp64. On an H100 a single forward fits; chunk_seqs>0
    accumulates over sequence chunks for tight-memory boxes (each supported loss
    is a MEAN over kept positions, so chunks are weighted by kept-token share)."""
    param = _get_param(model, param_path)
    param.requires_grad_(True)   # analysis needs grad even if frozen during training
    X, Y = get_batch()
    n_seq = X.shape[0]
    model.zero_grad(set_to_none=True)
    if chunk_seqs is None or chunk_seqs >= n_seq:
        _, loss = model(X, Y)
        loss.backward()
    else:
        tot = int((Y.view(-1) != -1).sum().item())
        for s in range(0, n_seq, chunk_seqs):
            e = min(s + chunk_seqs, n_seq)
            xb, yb = X[s:e], Y[s:e]
            kept = int((yb.view(-1) != -1).sum().item())
            _, loss = model(xb, yb)
            (loss * (kept / max(tot, 1))).backward()
    if param.grad is None:
        # param did not receive grad (e.g. detached/frozen graph) -> zeros,
        # which makes Adam/Muon degenerate to P=I on this layer.
        G = torch.zeros_like(param, dtype=torch.float64).cpu()
    else:
        G = param.grad.detach().clone().to(torch.float64).cpu()
    model.zero_grad(set_to_none=True)
    return G


# ---------------------------------------------------------------------------
# shared block-Hessian factor M and per-block scales, per layer
# ---------------------------------------------------------------------------
def _forward_chunks(model, get_batch, chunk_seqs):
    """Yield per-chunk (X, Y) so a forward hook can accumulate over the full
    batch without materializing all logits at once (vocab=10000 -> GBs)."""
    X, Y = get_batch()
    n_seq = X.shape[0]
    step = n_seq if (chunk_seqs is None or chunk_seqs >= n_seq) else chunk_seqs
    for s in range(0, n_seq, step):
        yield X[s:s + step], Y[s:s + step]


def block_hessian_factor(model, get_batch, layer, device, chunk_seqs=None):
    """Return (M, scales): every block's raw Hessian is  H_k = scales[k] * M.

    lm_head   -> M = (1/N) sum_t x_t x_t^T,   scales = ones(C)   (class-shared)
    embedding -> M = W^T W (W=lm_head.weight), scales = N_v/N     (0-layer GN)
    Both hold whether or not embedding is frozen; embedding form is the EXACT
    Hessian only for n_layer==0 (logits linear in the embedding row)."""
    if layer == "lm_head":
        head = model.lm_head
        d = head.in_features
        lt = model.config.loss_type
        c = 1.0 if lt == "mse_rep" else (2.0 / head.out_features if lt == "mse" else None)
        if c is None:
            raise NotImplementedError(
                "lm_head preconditioned analysis implemented for mse/mse_rep only "
                "(ce block is class-dependent, no shared factor).")
        Gram = torch.zeros((d, d), dtype=torch.float64, device=device)
        n_tok = 0
        cap = {}
        h = head.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach()))
        model.eval()
        with torch.no_grad():
            for xb, yb in _forward_chunks(model, get_batch, chunk_seqs):
                model(xb, yb)
                feat = cap["x"].reshape(-1, d).to(torch.float64)
                Gram += feat.t() @ feat
                n_tok += feat.shape[0]
        h.remove()
        M = c * (Gram / max(1, n_tok))
        scales = torch.ones(head.out_features, dtype=torch.float64, device=device)
        return M, scales

    elif layer == "embedding":
        assert model.config.n_layer == 0, \
            "embedding GN block is exact only for n_layer==0"
        W = model.lm_head.weight.detach().to(torch.float64).to(device)   # (C,d)
        M = W.t() @ W                                                    # (d,d)
        emb = model.tok_emb
        V = emb.num_embeddings
        cnt = torch.zeros(V, dtype=torch.float64, device=device)
        cap = {}
        h = emb.register_forward_hook(lambda m, i, o: cap.__setitem__("ids", i[0].detach()))
        model.eval()
        with torch.no_grad():
            for xb, yb in _forward_chunks(model, get_batch, chunk_seqs):
                model(xb, yb)
                ids = cap["ids"].reshape(-1)
                cnt.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float64))
        h.remove()
        scales = cnt / max(1, int(cnt.sum().item()))
        return M, scales

    raise ValueError(layer)


# ---------------------------------------------------------------------------
# preconditioners
# ---------------------------------------------------------------------------
def adam_replay_ema(run_dir, tags, exp_dict, param_path, device, beta2, chunk_seqs):
    """Replay Adam's second moment v_t from init to each tag. Returns {tag: v_hat}
    where v_hat is (n_rows, d) numpy (bias-corrected)."""
    v = None
    t = 0
    out = {}
    for tag in tags:
        model, _ = load_model_and_ckpt(run_dir, tag, device)
        if model is None:
            continue
        gb = get_dataloader_one_batch(exp_dict, model.config.block_size, device)
        G = compute_gradient_fullbatch(model, gb, param_path, chunk_seqs)  # cpu fp64
        if v is None:
            v = torch.zeros_like(G)
        t += 1
        v.mul_(beta2).addcmul_(G, G, value=1.0 - beta2)
        out[tag] = (v / (1.0 - beta2 ** t)).numpy()
        print(f"  Adam EMA {tag}: t={t} ||v_hat||={np.linalg.norm(out[tag]):.3e}", flush=True)
        del model
    return out


def muon_Pinv_sqrt(G, ns_steps, device):
    """NS5-exact right preconditioner P^{-1/2}, P=(G^T G)^{1/2}=NS5(G)^T G.

    G: (n_rows, d) torch fp64. Returns (d,d) torch fp64 on device."""
    G = G.to(device)
    if G.norm() < 1e-12:
        return torch.eye(G.size(-1), dtype=torch.float64, device=device)
    msign = _zeropower_via_newtonschulz5(G, steps=ns_steps).to(torch.float64)
    P = msign.t() @ G                       # ~ (G^T G)^{1/2}, exact in ideal NS
    P = 0.5 * (P + P.t())                    # symmetrize away NS approximation
    w, Vv = torch.linalg.eigh(P)
    w = w.clamp_min(1e-12)
    return (Vv / w.sqrt().unsqueeze(0)) @ Vv.t()


# ---------------------------------------------------------------------------
# spectrum summaries
# ---------------------------------------------------------------------------
def spectral_entropy(eigs):
    eigs = np.clip(np.asarray(eigs, float), 0, None)
    s = eigs.sum()
    if s <= 0:
        return 0.0
    lam = eigs / s
    lam = lam[lam > 1e-15]
    if lam.size < 2:
        return 0.0
    return float(-np.sum(lam * np.log(lam)) / np.log(lam.size))


def eigvalsh(mat_t):
    return torch.linalg.eigvalsh(mat_t).cpu().numpy()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--layer", required=True, choices=["lm_head", "embedding"])
    ap.add_argument("--optim", required=True, choices=["sgd", "adam", "muon"])
    ap.add_argument("--device", default=None, help="cuda / cpu (auto if unset)")
    ap.add_argument("--max_blocks", type=int, default=256)
    ap.add_argument("--subsample_eigh", type=int, default=16,
                    help="Adam: #blocks to eigh for within-block Y (others reuse trace-only X)")
    ap.add_argument("--chunk_seqs", type=int, default=None,
                    help="accumulate gradient over sequence chunks (tight memory)")
    ap.add_argument("--ns_steps", type=int, default=5)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    param_path = LAYER_PARAM[args.layer]
    print(f"device={device} layer={args.layer} optim={args.optim}", flush=True)

    exp_dict = None
    for t in TAG_ORDER:
        _, exp_dict = load_model_and_ckpt(args.run_dir, t, device)
        if exp_dict is not None:
            break
    if exp_dict is None:
        raise SystemExit(f"no checkpoints in {args.run_dir}")
    print(f"run optimizer (as trained): {exp_dict['optim']['name']}", flush=True)

    out_dir = os.path.join(args.run_dir, "precond_hessian", f"{args.layer}_{args.optim}")
    os.makedirs(out_dir, exist_ok=True)

    v_dict = {}
    if args.optim == "adam":
        beta2 = exp_dict["optim"].get("betas", (0.9, 0.95))[1]
        print(f"Replaying Adam EMA (beta2={beta2}) on {param_path} ...", flush=True)
        v_dict = adam_replay_ema(args.run_dir, TAG_ORDER, exp_dict, param_path,
                                 device, beta2, args.chunk_seqs)

    rows = []
    for tag in TAG_ORDER:
        model, _ = load_model_and_ckpt(args.run_dir, tag, device)
        if model is None:
            continue
        gb = get_dataloader_one_batch(exp_dict, model.config.block_size, device)

        M, scales = block_hessian_factor(model, gb, args.layer, device, args.chunk_seqs)
        d = M.shape[0]
        n_blocks = min(args.max_blocks, scales.numel())
        scales = scales[:n_blocks]
        scales_np = scales.cpu().numpy()

        if args.optim == "sgd":
            # H_k = scale_k * M -> eig = scale_k * eig(M); specH scale-invariant.
            eM = eigvalsh(M)
            block_mean = scales_np * eM.mean()
            Y = spectral_entropy(eM)               # identical for every block
            eigs_store = eM                        # shared spectrum

        elif args.optim == "muon":
            G = compute_gradient_fullbatch(model, gb, param_path, args.chunk_seqs)
            Pis = muon_Pinv_sqrt(G[:n_blocks], args.ns_steps, device)   # (d,d)
            Mpc = Pis @ M @ Pis
            eMpc = eigvalsh(Mpc)
            block_mean = scales_np * eMpc.mean()   # shared P -> X = std(log scale)
            Y = spectral_entropy(eMpc)
            eigs_store = eMpc

        else:  # adam: D_k differs per block; eig(scale_k * D_k M D_k)
            v_hat = v_dict[tag][:n_blocks]                    # (nb, d)
            # Adam preconditioner P_k = diag(sqrt(v_k)); the symmetric
            # preconditioned Hessian is P_k^{-1/2} M P_k^{-1/2}, so the diagonal
            # factor is D_k = P_k^{-1/2} = diag(v_k^{-1/4}).
            p = 1.0 / (v_hat ** 0.25 + EPS)                   # D_k diagonal (nb,d)
            Hdiag = torch.diagonal(M).cpu().numpy()           # (d,)
            # X uses ALL blocks: mean eig = trace/d = scale_k * mean_i p_{k,i}^2 H_ii
            block_mean = scales_np * (p ** 2 * Hdiag[None, :]).mean(axis=1)
            # Y from subsampled per-block eigh (specH is scale-invariant)
            n_e = min(args.subsample_eigh, n_blocks)
            specs = np.empty(n_e)
            eig_sub = np.empty((n_e, d))
            for k in range(n_e):
                pk = torch.as_tensor(p[k], dtype=torch.float64, device=device)
                Hk = pk[:, None] * M * pk[None, :]
                ek = eigvalsh(Hk)
                eig_sub[k] = ek
                specs[k] = spectral_entropy(ek)
                del Hk
            Y = float(specs.mean())
            eigs_store = eig_sub

        pos = block_mean[block_mean > 0]
        X = float(np.std(np.log10(pos))) if pos.size > 1 else 0.0

        np.save(os.path.join(out_dir, f"{tag}_block_mean.npy"), block_mean)
        np.save(os.path.join(out_dir, f"{tag}_eigs.npy"), eigs_store)
        summ = {"tag": tag, "layer": args.layer, "optim": args.optim,
                "n_blocks": int(n_blocks), "X_between": X, "Y_within_specH": float(Y)}
        with open(os.path.join(out_dir, f"{tag}_summary.json"), "w") as f:
            json.dump(summ, f, indent=2)
        rows.append(summ)
        print(f"  {tag:5s} X={X:.4f} Y={Y:.4f}", flush=True)
        del model, M

    with open(os.path.join(out_dir, "all_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nDone -> {out_dir}/all_summary.json", flush=True)


if __name__ == "__main__":
    main()
