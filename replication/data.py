"""
data.py
=======
Load a fixed dual-stream token dataset (train_x/train_y .bin + meta.pkl) and
reduce it to unique (x, y) pair counts.

For the linear bigram model logits = e_x^T E W, the per-position loss
0.5 * || e_x^T E W - e_y^T ||^2 depends on the position ONLY through the pair
(x, y), so the full-batch objective over N tokens is exactly

    L(W) = sum_{pairs} (c_xy / N) * 0.5 * || e_x^T E W - e_y^T ||^2

-- unique pair counts are a lossless compression of the dataset for this model.
"""

import os
import pickle

import numpy as np
import torch


def load_pairs(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    V = int(meta["vocab_size"])
    pi = np.asarray(meta["pi"], dtype=np.float64)   # analytic zipf, rank order
    label_mode = meta.get("label_mode", "shift")
    del meta                                        # bigram meta holds an 800MB P

    x = np.fromfile(os.path.join(data_dir, "train_x.bin"),
                    dtype=np.uint16).astype(np.int64)
    y = np.fromfile(os.path.join(data_dir, "train_y.bin"),
                    dtype=np.uint16).astype(np.int64)
    assert len(x) == len(y) and x.max() < V and y.max() < V

    pair_ids = x * V + y
    uniq, counts = np.unique(pair_ids, return_counts=True)
    return {
        "V": V,
        "n_tokens": len(x),
        "pair_x": uniq // V,
        "pair_y": uniq % V,
        "pair_c": counts.astype(np.float64),
        "pi": pi,
        "label_mode": label_mode,
        "name": os.path.basename(os.path.normpath(data_dir)),
    }


class DatasetBigramObjective:
    """Deterministic full-batch objective over a fixed token dataset:

        L(W) = (1/N) sum_t 0.5 * || e_{x_t}^T E W - e_{y_t}^T ||^2
             = sum_x l_x(W),   l_x = 0.5*rho_x*||R_x||^2 - sum_y w_xy R_xy
                                     + 0.5*rho_x,   R = E W

    where w_xy = c_xy / N (sums to 1) and rho_x = sum_y w_xy is the empirical
    input-token frequency.  The Hessian is (E^T diag(rho) E) (x) I -- constant,
    eigenvalues exactly {rho_x} -- so the lr regimes match the population
    problem (whose eigenvalues are the analytic pi).

    The gradient is analytic: grad = E^T diag(rho) E W - E^T T with
    T[x, y] = w_xy; E^T T is constant and precomputed once.

    Because E is orthogonal, E W can represent ANY matrix, so the per-x floor
    (irreducible loss, reached at R_x = empirical P(y|x)) is

        floor_x = 0.5 * (rho_x - ||T[x, :]||^2 / rho_x)

    (identically 0 for identity data).  Excess loss = l_x - floor_x -> 0.
    """

    def __init__(self, pairs, E, device, dtype):
        V = pairs["V"]
        self.V = V
        self.name = pairs["name"]
        w = torch.from_numpy(pairs["pair_c"] / pairs["n_tokens"])
        px = torch.from_numpy(pairs["pair_x"])
        py = torch.from_numpy(pairs["pair_y"])

        rho = torch.zeros(V, dtype=torch.float64)
        rho.index_add_(0, px, w)
        # ||T[x,:]||^2 accumulated from the sparse pair weights
        t_sq = torch.zeros(V, dtype=torch.float64)
        t_sq.index_add_(0, px, w * w)
        floor = torch.where(rho > 0, 0.5 * (rho - t_sq / rho.clamp(min=1e-300)),
                            torch.zeros_like(rho))

        self.rho = rho.to(device, dtype)
        self.floor = floor.to(device, dtype)
        self.px = px.to(device)
        self.py = py.to(device)
        self.w = w.to(device, dtype)
        self.E = E                                       # (V,V) on device
        T = torch.zeros(V, V, device=device, dtype=dtype)
        T[self.px, self.py] = self.w
        self.ETT = E.T @ T                               # constant part of grad
        del T

    def per_x_losses(self, W):
        """l_x vector (count-weighted, sums to the total objective)."""
        R = self.E @ W
        cross = torch.zeros(self.V, device=W.device, dtype=W.dtype)
        cross.index_add_(0, self.px, self.w * R[self.px, self.py])
        return 0.5 * self.rho * (R * R).sum(dim=1) - cross + 0.5 * self.rho

    def grad(self, W):
        R = self.E @ W
        return self.E.T @ (self.rho.view(-1, 1) * R) - self.ETT
