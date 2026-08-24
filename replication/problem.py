"""
problem.py
==========
Faithful port of the reference "RotatedMatrixBigramProblem".

The problem is a purely ANALYTIC population objective -- no sampled dataset:

    E   : fixed dim x dim orthogonal matrix (QR of seeded Gaussian), the
          frozen "embedding".
    W   : dim x dim trainable parameter (the "lm_head"), initialised to zeros.
    pi  : Zipf class weights, pi_i proportional to 1/rank (s = 1), exact.
    loss: f(W) = sum_i pi_i * 0.5 * || e_i^T E W - e_i^T ||^2
          i.e. pi-weighted squared error of the logits E W against the
          identity target (y == x), with a 0.5 * per-class SUM convention
          (NOT a mean over classes).

Because E is orthogonal, the Hessian of f is (E^T diag(pi) E) (x) I, whose
eigenvalues are exactly the Zipf weights {pi_i}, each with multiplicity dim.

Differences from the reference snippet: `@attrs.frozen` is replaced by the
stdlib `@dataclass(frozen=True)` (no behaviour change), and the analytic
gradient is verified against autograd in tests/smoke mode.
"""

from dataclasses import dataclass

import torch

E_cache = {}


@dataclass(frozen=True)
class RotatedMatrixBigramProblem:
    dim: int = 1
    # "orthogonal": the reference E = QR(seeded Gaussian) -- rows orthonormal,
    #     so classes map to orthogonal directions and the Hessian eigenvalues
    #     are exactly pi.
    # "gaussian": E ~ N(0, embed_std^2) entrywise, matching the frozen
    #     tok_emb init of the toy_models REP-* runs (initG02 = std 0.2).  E is
    #     still invertible a.s. (so the loss floor is unchanged), but its
    #     rows are no longer orthonormal: the Hessian E^T diag(pi) E mixes
    #     classes across shared feature directions, which is the setting the
    #     toy_models runs actually train.
    embed_init: str = "orthogonal"
    embed_std: float = 0.2

    def cache_key(self):
        return (self.dim, self.embed_init, self.embed_std)

    def _get_E(self) -> torch.Tensor:
        key = self.cache_key()
        if key not in E_cache:
            g = torch.Generator(device="cpu")
            g.manual_seed(self.dim)
            A = torch.randn(self.dim, self.dim, generator=g)
            if self.embed_init == "orthogonal":
                E_cache[key], _ = torch.linalg.qr(A)
            elif self.embed_init == "gaussian":
                E_cache[key] = A * self.embed_std
            else:
                raise ValueError(f"unknown embed_init {self.embed_init!r}")
        return E_cache[key]

    def get_initialization(self) -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.zeros(self.dim, self.dim),
                                  requires_grad=True)

    def _get_eigs(self, W: torch.Tensor) -> torch.Tensor:
        return 1 / torch.arange(1, self.dim + 1, device=W.device, dtype=W.dtype)

    def _get_probs(self, W: torch.Tensor) -> torch.Tensor:
        eigs = self._get_eigs(W)
        return eigs / eigs.sum()

    def _per_class_losses(self, W: torch.Tensor) -> torch.Tensor:
        E = self._get_E().to(W.device, W.dtype)
        EW = E @ W

        row_sq_norms = (EW ** 2).sum(dim=1)
        diag = torch.diagonal(EW)
        return 0.5 * (row_sq_norms - 2.0 * diag + 1.0)

    def f(self, W: torch.nn.Parameter):
        pi = self._get_probs(W)
        return (pi * self._per_class_losses(W)).sum()

    def full_grad(self, W: torch.Tensor) -> torch.Tensor:
        E = self._get_E().to(W.device, W.dtype)
        pi = self._get_probs(W)

        R = E @ W
        idx = torch.arange(self.dim, device=W.device)
        R[idx, idx] -= 1.0

        weighted_R = R * pi.view(-1, 1)
        grad = E.T @ weighted_R
        return grad

    # ---- convenience additions (not in the reference) --------------------- #
    def weighted_grad(self, W: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Gradient of sum_i w_i * per_class_loss_i(W) for arbitrary weights w.

        With w = pi this equals full_grad (population gradient); with w =
        empirical class frequencies of a sampled batch it is EXACTLY the
        mini-batch stochastic gradient of the mean per-sample loss, because
        the per-sample loss depends on the sample only through its class.
        """
        E = self._get_E().to(W.device, W.dtype)
        R = E @ W
        idx = torch.arange(self.dim, device=W.device)
        R[idx, idx] -= 1.0
        return E.T @ (R * w.view(-1, 1))

    def hessian_eigs(self, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """Exact Hessian spectrum: the pi_i values, each with multiplicity dim.

        Only valid for embed_init="orthogonal"; for a Gaussian E the Hessian is
        E^T diag(pi) E (x) I and its spectrum is not pi -- use
        hessian_lambda_max for the stability bound there.
        """
        assert self.embed_init == "orthogonal", self.embed_init
        W = torch.empty(0, device=device, dtype=dtype)
        return self._get_probs(W)

    def hessian_lambda_max(self, weights=None, iters=200) -> float:
        """lambda_max of E^T diag(w) E by power iteration (w defaults to pi).

        The Hessian of the objective is E^T diag(w) E (x) I, so this is the
        full Hessian's top eigenvalue and GD is stable for lr < 2/lambda_max.
        """
        E = self._get_E()
        w = (self._get_probs(E[:, :1]).to(E.dtype) if weights is None
             else weights.to(E.device, E.dtype))
        v = torch.randn(self.dim, device=E.device, dtype=E.dtype,
                        generator=torch.Generator(device=E.device).manual_seed(0))
        v /= v.norm()
        lam = 0.0
        for _ in range(iters):
            u = E.T @ (w.view(-1, 1) * (E @ v.view(-1, 1)))
            lam = float(u.norm())
            v = (u / max(lam, 1e-300)).view(-1)
        return lam
