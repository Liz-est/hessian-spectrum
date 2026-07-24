"""
transition.py
=============
Core mathematical machinery for the synthetic bigram language.

Two *orthogonal* knobs, decoupled by construction:

  1. pi (the stationary token-frequency distribution)   -> "balance vs imbalance"
     controls WHICH tokens are frequent/rare  -> class imbalance in the
     unembedding / softmax head of the model.

  2. predictability (a scalar in [0, 1])                -> "task difficulty"
     controls HOW predictable the next token is given the current one, WITHOUT
     changing pi at all -> how low the loss can go / how ill-conditioned the
     landscape is.

Why they are decoupled (the key trick)
--------------------------------------
The transition kernel is a mixture of two kernels that BOTH have the exact same
stationary distribution pi:

    P = (1 - a) * Pi_indep  +  a * B

  * Pi_indep[i, j] = pi[j]            (rank-1 "independence" kernel)
        - stationary distribution is exactly pi
        - each row equals pi  ->  maximal entropy  ->  next token is
          unpredictable beyond the marginal (hardest task).

  * B = Metropolis-Hastings kernel with a *concentrated* proposal
        - stationary distribution is exactly pi (MH guarantees this for ANY
          proposal, by reversibility)
        - rows are sharp (low entropy)  ->  next token is highly predictable
          (easiest task).

Because a convex combination of two kernels sharing stationary pi also has
stationary pi, the marginal token frequency stays EXACTLY pi for every value of
`a = predictability`.  So you can sweep difficulty while holding word frequency
fixed, or sweep word frequency while holding difficulty fixed -- clean
controlled experiments for Hessian analysis.

The order is fixed to bigram (first-order Markov) but the code is organised so a
higher-order n-gram sampler can be dropped in later (see `sample_sequence`).
"""

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Stationary token-frequency distribution  pi                              #
# --------------------------------------------------------------------------- #
def make_pi(vocab_size, kind="zipf", zipf_s=1.0, real_counts=None, eps=1e-12):
    """Build the target stationary distribution pi over `vocab_size` tokens.

    Parameters
    ----------
    vocab_size : int
    kind : {"uniform", "zipf", "real"}
        "uniform" -> perfectly balanced data.
        "zipf"    -> pi[r] proportional to 1 / (r + 1) ** zipf_s   (imbalanced).
        "real"    -> align to an empirical unigram distribution supplied via
                     `real_counts` (a 1-D array of counts, length vocab_size).
    zipf_s : float
        Zipf exponent.  0 -> uniform, 1 -> classic Zipf, >1 -> heavier imbalance.
    real_counts : array-like or None
        Empirical counts, required when kind == "real".

    Returns
    -------
    pi : np.ndarray, shape (vocab_size,), sums to 1, strictly positive.
    """
    if kind == "uniform":
        pi = np.ones(vocab_size, dtype=np.float64)
    elif kind == "zipf":
        ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
        pi = 1.0 / np.power(ranks, zipf_s)
    elif kind == "real":
        if real_counts is None:
            raise ValueError("kind='real' requires real_counts")
        pi = np.asarray(real_counts, dtype=np.float64).copy()
        if pi.shape[0] != vocab_size:
            raise ValueError(
                f"real_counts length {pi.shape[0]} != vocab_size {vocab_size}"
            )
    else:
        raise ValueError(f"unknown pi kind: {kind}")

    # Keep pi strictly positive so log(pi) and MH ratios are always finite.
    pi = pi + eps
    pi = pi / pi.sum()
    return pi


# --------------------------------------------------------------------------- #
# 2. Concentrated proposal kernel Q (gives the sharp / learnable structure)   #
# --------------------------------------------------------------------------- #
def make_proposal(vocab_size, bandwidth_frac=0.02, rng=None):
    """A concentrated proposal Q for Metropolis-Hastings.

    Q[i, j] is a (circular) Gaussian bump centred on i, so token i tends to be
    followed by tokens with nearby indices.  This injects *learnable* local
    structure whose sharpness is set by `bandwidth_frac`:

        small bandwidth -> very concentrated proposal -> sharp B -> easy task
        large bandwidth -> diffuse proposal          -> soft  B -> harder task

    The token index ordering is arbitrary (tokens have no natural order); it is
    merely a device to build a well-defined, reproducible, low-entropy kernel.

    Returns
    -------
    Q : np.ndarray, shape (vocab_size, vocab_size), rows sum to 1.
    """
    h = max(bandwidth_frac * vocab_size, 1e-6)
    idx = np.arange(vocab_size)
    # circular distance so the kernel is homogeneous and symmetric
    d = np.abs(idx[:, None] - idx[None, :])
    d = np.minimum(d, vocab_size - d)
    Q = np.exp(-0.5 * (d / h) ** 2)
    Q /= Q.sum(axis=1, keepdims=True)
    return Q


# --------------------------------------------------------------------------- #
# 3. Metropolis-Hastings kernel B with stationary distribution exactly pi     #
# --------------------------------------------------------------------------- #
def make_mh_kernel(pi, Q):
    """Metropolis-Hastings transition kernel that is reversible w.r.t. pi.

    B[i, j] = Q[i, j] * min(1, (pi[j] Q[j, i]) / (pi[i] Q[i, j]))   (j != i)
    B[i, i] = 1 - sum_{j != i} B[i, j]

    By detailed balance the stationary distribution is EXACTLY pi for any Q.
    """
    pi = np.asarray(pi, dtype=np.float64)
    V = pi.shape[0]

    # acceptance ratio  a[i,j] = min(1, (pi_j Q_ji) / (pi_i Q_ij))
    num = pi[None, :] * Q.T          # pi[j] * Q[j, i]
    den = pi[:, None] * Q            # pi[i] * Q[i, j]
    with np.errstate(divide="ignore", invalid="ignore"):
        accept = np.minimum(1.0, num / den)
    accept[den == 0.0] = 0.0         # never propose where Q is 0

    B = Q * accept
    np.fill_diagonal(B, 0.0)
    # remaining mass (rejections) stays on the diagonal -> self-loops
    diag = 1.0 - B.sum(axis=1)
    np.fill_diagonal(B, diag)
    return B


# --------------------------------------------------------------------------- #
# 4. Assemble the final transition matrix P                                   #
# --------------------------------------------------------------------------- #
def build_transition(pi, predictability=0.8, bandwidth_frac=0.02, rng=None):
    """Mixture kernel  P = (1 - a) * Pi_indep + a * B,  stationary = pi exactly.

    Parameters
    ----------
    pi : np.ndarray
        Target stationary distribution (from `make_pi`).
    predictability : float in [0, 1]
        The difficulty knob `a`, DECOUPLED from pi:
            0.0 -> P rows all equal pi  (max entropy, next token ~ marginal,
                   hardest: model can only learn the unigram frequencies).
            1.0 -> P = B  (sharp, low entropy, next token highly predictable,
                   easiest).
        Intermediate values interpolate continuously.
        NB: relation to the "temperature" framing -- high predictability = low
        temperature (sharp), low predictability = high temperature (diffuse).
    bandwidth_frac : float
        Structural sharpness of the learnable component B (secondary knob).

    Returns
    -------
    P : np.ndarray, shape (V, V), rows sum to 1, stationary distribution = pi.
    """
    assert 0.0 <= predictability <= 1.0, "predictability must be in [0, 1]"
    V = pi.shape[0]
    Pi_indep = np.broadcast_to(pi, (V, V))          # every row == pi
    B = make_mh_kernel(pi, make_proposal(V, bandwidth_frac, rng))
    P = (1.0 - predictability) * Pi_indep + predictability * B
    # numerical clean-up
    P = np.clip(P, 0.0, None)
    P /= P.sum(axis=1, keepdims=True)
    return P


# --------------------------------------------------------------------------- #
# 5. Diagnostics                                                              #
# --------------------------------------------------------------------------- #
def stationary_distribution(P, tol=1e-10, max_iter=100000):
    """Compute the stationary distribution of P by power iteration (for checks)."""
    V = P.shape[0]
    v = np.ones(V) / V
    for _ in range(max_iter):
        v_next = v @ P
        if np.abs(v_next - v).sum() < tol:
            v = v_next
            break
        v = v_next
    return v / v.sum()


def row_entropy(P):
    """Per-row Shannon entropy (nats).  Low = predictable, high = uncertain."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P), 0.0)
    return -(P * logP).sum(axis=1)


# --------------------------------------------------------------------------- #
# 6. Markov sampling                                                          #
# --------------------------------------------------------------------------- #
def sample_sequence(P, pi, length, rng):
    """Sample a length-`length` token sequence from the bigram chain.

    Starts from the stationary distribution pi so the whole sequence is
    stationary (no burn-in artefacts).  Uses the CDF-inversion trick with a
    single vector of uniforms for speed.

    NOTE (n-gram extension point): for a k-th order model this function would
    take a rank-(k+1) tensor `P[c_{t-k},...,c_{t-1}, :]` and condition on the
    last k tokens.  The rest of the pipeline (writing x/y, meta) is unchanged.
    """
    V = pi.shape[0]
    cdf = np.cumsum(P, axis=1)
    cdf[:, -1] = 1.0                      # guard against fp drift
    out = np.empty(length, dtype=np.int64)
    # initial token from the stationary distribution
    out[0] = rng.choice(V, p=pi)
    u = rng.random(length)
    cur = out[0]
    for t in range(1, length):
        cur = np.searchsorted(cdf[cur], u[t], side="right")
        if cur >= V:                      # numerical edge case
            cur = V - 1
        out[t] = cur
    return out


def sample_sequences_batch(P, pi, n_seqs, length, rng):
    """Sample `n_seqs` independent length-`length` sequences in lock-step.

    Statistically identical to calling `sample_sequence` n_seqs times: every
    row starts from the stationary distribution pi and then follows P, and the
    rows are mutually independent.  The difference is purely computational --
    all rows advance together, one vectorised CDF lookup per time step, instead
    of a Python-level loop per token.

    The batched inverse-CDF lookup uses a row-offset trick: shifting row i of
    the per-state CDF by +i makes the flattened (V*V,) table globally
    non-decreasing (row i spans (i, i+1]), so a single np.searchsorted with
    queries `cur + u` lands inside row `cur` at exactly the index the scalar
    version would pick.

    Returns
    -------
    out : np.ndarray, shape (n_seqs, length), dtype int64
    """
    V = pi.shape[0]
    cdf = np.cumsum(P, axis=1)
    cdf[:, -1] = 1.0                      # guard against fp drift
    offset_cdf = (cdf + np.arange(V)[:, None]).ravel()

    out = np.empty((n_seqs, length), dtype=np.int64)
    cur = rng.choice(V, p=pi, size=n_seqs)
    out[:, 0] = cur
    u = rng.random((n_seqs, length))
    for t in range(1, length):
        g = np.searchsorted(offset_cdf, cur + u[:, t], side="right")
        cur = np.minimum(g - cur * V, V - 1)   # numerical edge case
        out[:, t] = cur
    return out


# --------------------------------------------------------------------------- #
# 7. Output coupling K(y | x): DIFFERENT marginals pi_x, pi_y, yet DEPENDENT   #
# --------------------------------------------------------------------------- #
def sinkhorn_coupling(pi_x, pi_y, A, n_iter=2000, tol=1e-12):
    """Scale a positive affinity A into a joint M with prescribed marginals.

    Solves for diagonal scalings u, v so that  M = diag(u) A diag(v)  lands in
    the transportation polytope U(pi_x, pi_y):

        M >= 0,   M @ 1 = pi_x   (row sums = input marginal),
                  M.T @ 1 = pi_y (col sums = output marginal).

    For any strictly positive A the Sinkhorn/RAS iteration converges to the
    unique such M (it is the I-projection of A onto U, i.e. the min-KL coupling
    with those marginals).  M keeps the *structure* of A (e.g. a near-diagonal
    bump => x tends to map to y of a nearby index) while matching both margins
    exactly -- this is what makes x and y dependent yet with the marginals we
    asked for.

    Returns
    -------
    M : np.ndarray, shape (V, V), rows sum to pi_x, cols sum to pi_y.
    """
    pi_x = np.asarray(pi_x, dtype=np.float64)
    pi_y = np.asarray(pi_y, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64).copy()
    A /= A.sum()                                  # normalise for stability
    V = pi_x.shape[0]
    u = np.ones(V)
    v = np.ones(V)
    for _ in range(n_iter):
        # row scaling: (u A v).sum(axis=1) == pi_x
        Av = A @ v
        u = pi_x / np.maximum(Av, 1e-300)
        # col scaling: (u A v).sum(axis=0) == pi_y
        uA = A.T @ u
        v_new = pi_y / np.maximum(uA, 1e-300)
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new
    M = (u[:, None] * A) * v[None, :]
    M /= M.sum()                                  # kill residual fp drift
    return M


def build_coupling_kernel(pi_x, pi_y, strength=1.0, bandwidth_frac=0.02,
                          rng=None):
    """Conditional kernel K(y | x) with col-marginal exactly pi_y for any strength.

    We want y | x drawn so that (a) the achieved output marginal is exactly
    pi_y = sum_i pi_x[i] K[i, :], and (b) a knob `strength` in [0, 1] moves from
    INDEPENDENT (y ignores x) to a STRUCTURED dependence.  Both are secured by a
    convex combination that mirrors build_transition:

        K = (1 - b) * (1 pi_y^T)  +  b * R,    b = strength,
        with R a row-stochastic kernel whose col-marginal under pi_x is pi_y:
        R[i, j] = M[i, j] / pi_x[i],   M = sinkhorn_coupling(pi_x, pi_y, A).

    Because pi_x^T (1 pi_y^T) = pi_y and pi_x^T R = (1^T M) = pi_y, we get
    pi_x^T K = (1 - b) pi_y + b pi_y = pi_y for EVERY b -- the output marginal is
    held fixed while b tunes how much y depends on x.

        b = 0  -> K rows all equal pi_y  -> y independent of x (== 'independent'
                  mode, but with x/y still allowed different marginals).
        b = 1  -> K = R, the structured (near-diagonal) coupling: given x = i,
                  y concentrates on output tokens with nearby index.

    The affinity A is the same circular-Gaussian bump used for the input kernel
    (bandwidth_frac controls sharpness), so "structure" here means x maps to y of
    a similar index -- a clean, reproducible, low-entropy dependence.

    Returns
    -------
    K : np.ndarray, shape (V, V), row-stochastic (K @ 1 == 1),
        with pi_x^T K == pi_y.
    """
    assert 0.0 <= strength <= 1.0, "coupling strength must be in [0, 1]"
    pi_x = np.asarray(pi_x, dtype=np.float64)
    pi_y = np.asarray(pi_y, dtype=np.float64)
    V = pi_x.shape[0]

    A = make_proposal(V, bandwidth_frac, rng)          # positive bump affinity
    M = sinkhorn_coupling(pi_x, pi_y, A)               # joint with both margins
    R = M / np.maximum(pi_x[:, None], 1e-300)          # condition: R = p(y | x)
    R /= R.sum(axis=1, keepdims=True)                  # clean fp drift

    indep = np.broadcast_to(pi_y, (V, V))              # rows all equal pi_y
    K = (1.0 - strength) * indep + strength * R
    K = np.clip(K, 0.0, None)
    K /= K.sum(axis=1, keepdims=True)
    return K


def output_marginal(pi_x, K):
    """Achieved output marginal  pi_y_hat = pi_x^T K  (for diagnostics)."""
    return np.asarray(pi_x, dtype=np.float64) @ np.asarray(K, dtype=np.float64)


def sample_y_given_x(K, x_flat, rng):
    """Sample y_t ~ K(x_t, :) for a flat array of input tokens x_flat.

    Vectorised inverse-CDF with the same row-offset trick as
    sample_sequences_batch: peak memory is O(len(x_flat) + V*V) rather than
    materialising a (len(x_flat), V) probability table.

    Returns
    -------
    y : np.ndarray, same shape as x_flat, dtype int64.
    """
    V = K.shape[0]
    cdf = np.cumsum(K, axis=1)
    cdf[:, -1] = 1.0
    offset_cdf = (cdf + np.arange(V)[:, None]).ravel()
    x = np.asarray(x_flat, dtype=np.int64)
    u = rng.random(x.shape)
    g = np.searchsorted(offset_cdf, x + u, side="right")
    y = np.minimum(g - x * V, V - 1)
    return y
