"""
presets.py
==========
Run configurations for the RotatedMatrixBigramProblem replication.

The reference snippet fixes the PROBLEM (orthogonal E seeded by dim, W = 0
init, exact Zipf-s1 class weights, 0.5 * per-class-sum squared error) but does
NOT specify the optimizer / lr / schedule.  Optimizer settings below follow
the conventions of the earlier toy_models REP-* attempts (Adam betas=(0.0,
0.999), eps=1e-8, plain SGD, constant lr, 500 iters) with lr grids re-scaled
for THIS loss normalisation:

  * the loss here is a pi-weighted 0.5 * SUM over classes (not a mean over
    batch x classes), so gradients are ~V/2 = 5000x larger than the
    toy_models mse0 runs -- lrs are chosen for this problem, not copied.
  * the Hessian is constant with eigenvalues exactly pi_i; for dim=10000,
    lambda_max = pi_1 = 1/H(10000) ~= 0.102, so plain GD is stable for
    lr < 2/0.102 ~= 19.6.  The SGD grid stays well inside that.
"""

from dataclasses import dataclass, field

# fixed on-disk datasets for grad_mode="dataset" (dual-stream .bin + meta.pkl)
DATA_ROOT = "/data/250010020/hessian-spectrum/data"
DATASETS = {
    # identity task: y == x, x ~ zipf(s=1), V=10000, 100k train tokens
    "identity": f"{DATA_ROOT}/synth_identity_zipf_s1_V10000",
    # bigram shift task: y ~ P(.|x), zipf(s=1) marginal, V=10000, 100k tokens
    "bigram": f"{DATA_ROOT}/synth_zipf_imbalanced_s1_V10000",
}


@dataclass(frozen=True)
class RunConfig:
    name: str
    dim: int = 10000
    # frozen embedding E: "orthogonal" (the reference) or "gaussian" (entrywise
    # N(0, embed_std^2), matching the toy_models REP-* frozen tok_emb initG02).
    # Gaussian rows have norm ~ embed_std*sqrt(dim) = 20, so the Hessian
    # E^T diag(rho) E is ~400x larger (lambda_max 40.1 vs 0.0988 on the bigram
    # data) -- the SGD lr grid must shrink accordingly.
    embed_init: str = "orthogonal"
    embed_std: float = 0.2
    optimizer: str = "adam"            # "adam" | "sgd"
    lr: float = 1e-3
    betas: tuple = (0.0, 0.999)        # adam only (matches earlier REP-* runs)
    eps: float = 1e-8                  # adam only
    momentum: float = 0.0              # sgd only
    weight_decay: float = 0.0
    max_iters: int = 500
    dtype: str = "float32"
    # --- gradient mode ----------------------------------------------------
    # "population": deterministic full_grad with exact pi weights (the
    #     reference setting -- equivalent to infinite data).
    # "stochastic": each step samples batch_size i.i.d. classes x ~ pi
    #     (identity data y == x) and uses the empirical-frequency gradient --
    #     the exact mini-batch gradient of the mean per-sample loss.
    # "dataset": deterministic FULL-BATCH gradient over a fixed on-disk token
    #     dataset (see DATASETS); same 0.5*per-position-sum loss, same
    #     orthogonal E, W = 0 init.  Hessian eigs = empirical input-token
    #     frequencies rho ~= pi, so lr regimes match the population problem.
    grad_mode: str = "population"
    dataset: str = ""                  # DATASETS key, grad_mode="dataset" only
    batch_size: int = 100_000          # stochastic only (matches the 100k-token
                                       # full-batch setup of the earlier runs)
    data_seed: int = 1337              # stochastic only
    # fractions of max_iters at which the full per-class loss vector is saved
    ckpt_fracs: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    n_freq_groups: int = 10            # equal-mass pi groups for the plot
    save_final_W: bool = False         # W is dim^2 floats (~400 MB); off by default


def _adam(lr, mode="population"):
    tag = f"{lr:.0e}".replace("e-0", "e-")
    suffix = "" if mode == "population" else "-stoch"
    return RunConfig(name=f"rep-identity-adam-lr{tag}{suffix}",
                     optimizer="adam", lr=lr, grad_mode=mode)


def _sgd(lr, mode="population"):
    tag = str(lr).replace(".", "p")
    suffix = "" if mode == "population" else "-stoch"
    return RunConfig(name=f"rep-identity-sgd-lr{tag}{suffix}",
                     optimizer="sgd", lr=lr, grad_mode=mode)


def _data(ds, opt, lr):
    tag = (f"{lr:.0e}".replace("e-0", "e-") if opt == "adam"
           else str(lr).replace(".", "p"))
    return RunConfig(name=f"rep-data-{ds}-{opt}-lr{tag}",
                     optimizer=opt, lr=lr, grad_mode="dataset", dataset=ds)


# first sweep showed adam spikes/diverges for lr >= 1e-3 (constant-Hessian
# quadratic: as the gradient decays, v shrinks and the effective step lr/sqrt(v)
# grows -> edge-of-stability oscillation, worse with beta1=0); extended the
# grid below 3e-4, which was the only near-stable setting.
_LR_ADAM = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
_LR_SGD = [1.0, 5.0, 10.0, 15.0]

# dataset-mode grids (Hessian eigs = empirical rho ~= pi, so same regimes as
# population): adam centred on the stable 3e-5..1e-4 band found above; sgd
# keeps the monotone 1..15 band.
_LR_ADAM_DATA = [3e-4, 1e-4, 3e-5, 1e-5]
_LR_SGD_DATA = [1.0, 5.0, 10.0, 15.0]


def _gauss(ds, opt, lr, iters=500):
    """dataset-mode run with a Gaussian (non-orthogonal) frozen embedding."""
    tag = (f"{lr:.0e}".replace("e-0", "e-") if opt == "adam"
           else str(lr).replace(".", "p"))
    suffix = "" if iters == 500 else f"-it{iters}"
    return RunConfig(name=f"rep-gauss-{ds}-{opt}-lr{tag}{suffix}",
                     embed_init="gaussian", embed_std=0.2,
                     optimizer=opt, lr=lr, grad_mode="dataset", dataset=ds,
                     max_iters=iters)


# Gaussian-E grids on the bigram data.  lambda_max = 40.1 (power iteration on
# E^T diag(rho) E), so GD is stable only for lr < 2/40.1 = 0.0499: the SGD grid
# spans 3e-4..4e-2, i.e. up to ~0.8x the stability limit, with ~2x spacing.
# Adam's stable band scales differently (its step is lr * sign-like, not
# lr * grad), and the toy_models runs on this same data found the clean band at
# 6e-6..1e-5 with spikes by 6e-5 -- the grid brackets that by a decade either
# side since here the loss convention is V/2 = 5000x larger.
_LR_SGD_GAUSS = [3e-4, 1e-3, 3e-3, 6e-3, 1.2e-2, 2.4e-2, 4e-2,
                 # refinement: the first sweep was monotone in lr all the way to
                 # 4e-2, so the optimum sits at the stability boundary 0.0499
                 4.5e-2, 4.8e-2]
_LR_ADAM_GAUSS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
                  # refinement around the 3e-6 optimum (1e-5 already lets G0
                  # oscillate, 3e-5 spikes on 60 of 500 iters)
                  2e-6, 5e-6, 7e-6]

PRESETS = {
    c.name: c
    for c in
    # population: deterministic full grad, exact pi (the reference setting)
    [_adam(lr) for lr in _LR_ADAM] + [_sgd(lr) for lr in _LR_SGD]
    # stochastic: fresh 100k-sample batch each step, empirical-freq gradient
    + [_adam(lr, "stochastic") for lr in _LR_ADAM]
    + [_sgd(lr, "stochastic") for lr in _LR_SGD]
    # dataset: full-batch on the fixed on-disk datasets, sgd + adam lr grids
    + [_data(ds, "adam", lr) for ds in DATASETS for lr in _LR_ADAM_DATA]
    + [_data(ds, "sgd", lr) for ds in DATASETS for lr in _LR_SGD_DATA]
    # gaussian-E: same bigram data / loss / init, non-orthogonal frozen E
    + [_gauss("bigram", "sgd", lr) for lr in _LR_SGD_GAUSS]
    + [_gauss("bigram", "adam", lr) for lr in _LR_ADAM_GAUSS]
    # 4x-longer runs at the best lr of each family: separates "the tail groups
    # are structurally slow" from "500 iters is just not enough"
    + [_gauss("bigram", "sgd", 4.8e-2, iters=2000),
       _gauss("bigram", "adam", 3e-6, iters=2000),
       _gauss("bigram", "adam", 5e-6, iters=2000)]
    # tiny local sanity checks (CPU-friendly)
    + [RunConfig(name="rep-identity-smoke", dim=256, optimizer="adam",
                 lr=1e-3, max_iters=50),
       RunConfig(name="rep-identity-smoke-stoch", dim=256, optimizer="adam",
                 lr=1e-3, max_iters=50, grad_mode="stochastic",
                 batch_size=4096)]
}


def load(name: str) -> RunConfig:
    if name not in PRESETS:
        raise KeyError(
            f"unknown preset '{name}'. Available:\n  " + "\n  ".join(PRESETS))
    return PRESETS[name]
