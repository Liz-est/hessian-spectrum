"""
Exact per-unit Hessian analysis for vanilla_transformer (single-layer vanilla decoder).

Because the model is tiny we do NOT use stochastic Lanczos quadrature (SLQ).
Instead we form each *per-unit* Hessian / Gauss-Newton block exactly and
eigendecompose it (torch.linalg.eigvalsh, on GPU when available).

The "unit" — i.e. how a weight matrix is cut into blocks — depends on the layer,
per the experiment spec:

  * Q, K  (attn.wq / attn.wk)        -> block PER ATTENTION HEAD
  * V, attn.proj, mlp.fc, mlp.proj   -> block PER OUTPUT NEURON
  * embedding, lm_head               -> block PER TOKEN

Math per block kind
-------------------
Per output neuron (Linear y = W x, row w_i, y_i = w_i . x):
    H_i = (1/N) sum_t s_{i,t} x_t x_t^T           (in_dim x in_dim)
  with s_{i,t} = (dL/dy_{i,t})^2  (empirical-Fisher / Gauss-Newton curvature),
  EXACT for the fact that y_i is linear in w_i.

Per attention head (head h owns rows [h*hd:(h+1)*hd] of W, output y_h in R^hd,
flattened weight vec(W_h) in R^{hd*d}, per-token gradient vec = x_t ⊗ g_{h,t}):
    H_h = (1/N) sum_t (x_t x_t^T) ⊗ (g_{h,t} g_{h,t}^T)   ((hd*d) x (hd*d))
  formed as U^T U / N with rows u_t = x_t ⊗ g_{h,t}  (empirical Fisher, exact).

Per token, lm_head (CE, exact):  output neuron k == vocab token k,
    H_k = (1/N) sum_t p_{k,t}(1-p_{k,t}) x_t x_t^T       (d x d)
  s_{k,t} = p_{k,t}(1-p_{k,t}) is the exact CE curvature (matches the vision
  code's ce_last_layer_hessian_blocks).

Per token, lm_head (MSE-vs-one-hot, exact):  loss = mean over N*C of
(z-y)^2, so d^2 L / d z_k^2 = 2/C is constant and class-decoupled, giving
    H_k = (2/C) * (1/N) sum_t x_t x_t^T                  (d x d)
  the SAME block for every class k -- MSE removes the class-frequency-driven
  curvature heterogeneity that CE's p_k(1-p_k) weight introduces. C is the full
  vocab size (every class contributes to the mean, even un-selected ones).

Per token, embedding (Fisher):  token id v owns embedding row e_v in R^d,
    H_v = (1/N) sum_{t: x_t=v} g_t g_t^T               (d x d)
  g_t = dL/d(embedding output at position t); accumulated over the positions
  where token v actually appears. The denominator is the GLOBAL token count N
  (all positions in the analysis batches), NOT the per-token count N_v -- this
  is the true diagonal block of the *mean* training loss, whose magnitude scales
  with frequency as N_v/N (frequent tokens -> larger curvature blocks). Dividing
  by N_v instead would give the per-occurrence average curvature, a
  frequency-controlled probe that erases exactly the frequency heterogeneity the
  optimizer actually feels; it would also make embedding the only block not
  normalized like the others (neuron / lm_head both divide by the global N).

For each layer we get one block per unit, eigendecompose each, and:
  (1) pool eigenvalues over units  -> ESD (spectrum) plot;
  (2) turn each unit's spectrum into a log-eigenvalue probability histogram and
      compute pairwise Symmetric-KL and JS-distance matrices -> "hessian hetero".

Outputs land under files/toy_C/<tag>/ (tag in {init,p10,p25,p40,p50,p60,p75,p85,p100}).
"""

import os
import json

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-12


# ----------------------------------------------------------------------------
# spectra -> probability histograms (log-eigenvalue space) + pairwise distances
# ----------------------------------------------------------------------------
def spectra_to_prob(eig_rows, edges):
    """eig_rows: (n_units, k) eigenvalues. Return (n_units, n_bins) prob rows,
    each a normalized histogram of that unit's eigenvalues in log space."""
    P = []
    for row in eig_rows:
        vals = np.clip(np.asarray(row, float), 0.0, None)
        logs = np.log(vals + EPS)
        hist, _ = np.histogram(logs, bins=edges)
        hist = hist.astype(np.float64) + EPS          # Dirichlet smoothing
        P.append(hist / hist.sum())
    return np.vstack(P)


def common_log_edges(eig_rows, num_bins=64):
    allv = np.clip(np.concatenate([np.asarray(r, float).ravel() for r in eig_rows]), 0.0, None)
    z = np.log(allv + EPS)
    zmin, zmax = float(z.min()), float(z.max())
    if zmin == zmax:
        zmin, zmax = zmin - 1e-6, zmax + 1e-6
    return np.linspace(zmin, zmax, num_bins + 1)


def symmetric_kl(p, q):
    p = np.clip(p, EPS, None); q = np.clip(q, EPS, None)
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))


def js_distance(p, q):
    p = np.clip(p, EPS, None); q = np.clip(q, EPS, None)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(np.sqrt(max(js, 0.0)))


def pairwise_matrix(P, metric, device="cpu", chunk=64):
    """Pairwise Symmetric-KL ('skl') or JS-distance ('js') matrix for rows of P.

    Vectorized on `device` (GPU when available). P: (n, bins) probability rows.
    Returns an (n, n) numpy array with zero diagonal.
    """
    Pt = torch.as_tensor(np.clip(P, EPS, None), dtype=torch.float64, device=device)
    Pt = Pt / Pt.sum(dim=1, keepdim=True)
    logP = torch.log(Pt)
    n = Pt.shape[0]

    if metric == "skl":
        # KL(i||j) = a_i - sum_k P_ik log P_jk ;  a_i = sum_k P_ik log P_ik
        a = (Pt * logP).sum(dim=1)                       # (n,)
        M = Pt @ logP.t()                                # M[i,j] = sum_k P_ik logP_jk
        D = (a[:, None] + a[None, :]) - (M + M.t())      # symmetric KL
    else:  # js distance = sqrt( 0.5 KL(p||m) + 0.5 KL(q||m) )
        D = torch.zeros((n, n), dtype=torch.float64, device=device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            Pi = Pt[s:e][:, None, :]                      # (c,1,bins)
            Li = logP[s:e][:, None, :]
            Pj = Pt[None, :, :]                           # (1,n,bins)
            Lj = logP[None, :, :]
            m = 0.5 * (Pi + Pj)
            logm = torch.log(m)
            js = 0.5 * (Pi * (Li - logm)).sum(-1) + 0.5 * (Pj * (Lj - logm)).sum(-1)
            D[s:e] = torch.sqrt(js.clamp_min(0.0))
    D.fill_diagonal_(0.0)
    return D.cpu().numpy()


def hetero_mean(D):
    n = D.shape[0]
    idx = np.tril_indices(n, k=-1)
    return float(D[idx].mean()) if len(idx[0]) else 0.0


def cross_layer_matrices(save_dir, layer_names, num_bins=64, device="cpu"):
    """Cross-LAYER heterogeneity: pool each layer's per-unit eigenvalues into
    one spectrum, histogram all layers on shared log edges, and compute the
    pairwise layer-vs-layer Symmetric-KL / JS-distance matrices.

    Reads eigs_<layer>.npy from save_dir (skipping layers not yet computed) and
    saves hetero_layers_{skl,js}.npy next to them. Returns (layers, {"skl": D,
    "js": D}), or None if fewer than two layers have saved eigenvalues."""
    layers, rows = [], []
    for disp in layer_names:
        ef = os.path.join(save_dir, f"eigs_{disp}.npy")
        if os.path.exists(ef):
            layers.append(disp)
            rows.append(np.load(ef).ravel())
    if len(layers) < 2:
        return None
    edges = common_log_edges(rows, num_bins)
    P = spectra_to_prob(rows, edges)
    mats = {}
    for metric in ("skl", "js"):
        D = pairwise_matrix(P, metric, device=device)
        np.save(os.path.join(save_dir, f"hetero_layers_{metric}.npy"), D)
        mats[metric] = D
    return layers, mats


def _eigvalsh(mat_np, device="cpu"):
    """Symmetric eigenvalues of one block, on `device`. Returns numpy (dim,)."""
    t = torch.as_tensor(mat_np, dtype=torch.float64, device=device)
    ev = torch.linalg.eigvalsh(t)
    return ev.cpu().numpy()


# ----------------------------------------------------------------------------
# collect per-unit Hessian blocks over a set of batches
# ----------------------------------------------------------------------------
class NeuronHessian:
    """
    Accumulate exact per-unit blocks for each layer, then eigendecompose them.
    Blocks are formed on `device`; eigendecomposition also runs on `device`.
    """
    def __init__(self, model, get_batch, n_batches=20, device="cpu"):
        self.model = model
        self.get_batch = get_batch
        self.n_batches = n_batches
        self.device = device

    def _module(self, path):
        m = self.model
        for a in path.split("."):
            m = getattr(m, a)
        return m

    # ---- lm_head: exact CE block per token (== per output class) ----
    def last_layer_blocks(self, head_path="lm_head", max_classes=None,
                          class_ids=None, loss_type="ce"):
        """class_ids: explicit class ids to build blocks for (row order follows
        class_ids). When None, the first min(C, max_classes) ids are used.

        loss_type: "ce" -> exact softmax cross-entropy curvature
        p_k(1-p_k) x x^T (class-dependent). "mse" -> MSE-vs-one-hot curvature
        (2/C_full) x x^T, identical for every class (C_full = full vocab)."""
        head = self._module(head_path)
        d = head.in_features
        C_full = head.out_features
        C = C_full
        if class_ids is not None:
            ks = [int(k) for k in class_ids]
        else:
            if max_classes is not None:
                C = min(C, max_classes)
            ks = list(range(C))

        # MSE: every class shares one block H = (2/C_full) * mean_t x x^T, so we
        # accumulate the feature Gram once and skip the per-class softmax loop.
        if loss_type == "mse":
            Gram = torch.zeros((d, d), dtype=torch.float64, device=self.device)
            n_tok = 0
            captured = {}
            def hook(mod, inp, out):
                captured["x"] = inp[0].detach()
            h = head.register_forward_hook(hook)
            self.model.eval()
            with torch.no_grad():
                for _ in range(self.n_batches):
                    X, Y = self.get_batch()
                    self.model(X, Y)
                    feat = captured["x"].reshape(-1, d).to(torch.float64)     # (N,d)
                    Gram += feat.t() @ feat
                    n_tok += feat.shape[0]
            h.remove()
            block = (2.0 / C_full) * (Gram / max(1, n_tok))
            block_eigs = torch.linalg.eigvalsh(block).cpu().numpy()           # (d,)
            # identical spectrum for every selected class
            return np.tile(block_eigs, (len(ks), 1)).copy()

        Hsum = torch.zeros((len(ks), d, d), dtype=torch.float64, device=self.device)
        n_tok = 0

        captured = {}
        def hook(mod, inp, out):
            captured["x"] = inp[0].detach()
        h = head.register_forward_hook(hook)
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.n_batches):
                X, Y = self.get_batch()
                logits, _ = self.model(X, Y)
                feat = captured["x"].reshape(-1, d).to(torch.float64)         # (N,d)
                probs = F.softmax(logits.reshape(-1, logits.size(-1)), dim=-1)
                for slot, k in enumerate(ks):
                    pk = probs[:, k].to(torch.float64)
                    w = pk * (1.0 - pk)                                       # (N,)
                    Hsum[slot] += feat.t() @ (w.unsqueeze(1) * feat)
                n_tok += feat.shape[0]
        h.remove()
        Hsum /= max(1, n_tok)                                             # in place
        eigs = np.stack([torch.linalg.eigvalsh(Hsum[s]).cpu().numpy()
                         for s in range(len(ks))])
        return eigs

    # ---- embedding: Fisher block per token id ----
    def token_embedding_blocks(self, emb_path="tok_emb", max_tokens=None,
                               token_ids=None):
        """token_ids: explicit token ids to build blocks for (row order follows
        token_ids). When None, the first min(V, max_tokens) ids are used."""
        emb = self._module(emb_path)
        V_full = emb.num_embeddings
        d = emb.embedding_dim
        if token_ids is not None:
            sel = torch.as_tensor(np.asarray(token_ids), dtype=torch.long,
                                  device=self.device)
        else:
            V = V_full if max_tokens is None else min(V_full, max_tokens)
            sel = torch.arange(V, device=self.device)
        n_sel = sel.numel()
        # token id -> block slot (-1 = not selected)
        lut = torch.full((V_full,), -1, dtype=torch.long, device=self.device)
        lut[sel] = torch.arange(n_sel, device=self.device)
        Hsum = torch.zeros((n_sel, d, d), dtype=torch.float64, device=self.device)
        cnt = torch.zeros(n_sel, dtype=torch.float64, device=self.device)
        n_tok = 0                       # GLOBAL position count (all tokens seen)

        store = {}
        def fhook(mod, inp, out):
            store["ids"] = inp[0].detach()          # (B,T) token ids
            out.retain_grad()
            store["out"] = out                      # (B,T,d) embedding output
        h = emb.register_forward_hook(fhook)
        self.model.eval()
        for _ in range(self.n_batches):
            X, Y = self.get_batch()
            self.model.zero_grad(set_to_none=True)
            _, loss = self.model(X, Y)
            loss.backward()
            ids = store["ids"].reshape(-1)                                # (N,)
            g = store["out"].grad.reshape(-1, d).to(torch.float64)       # (N,d)
            n_tok += ids.numel()                     # count EVERY position, not
                                                     # just the selected tokens
            slot = lut[ids]
            mask = slot >= 0
            slot = slot[mask]; g = g[mask]
            # accumulate sum_t g_t g_t^T per selected token, computed as G_v^T G_v
            # over that token's rows -- identical to summing outer products but
            # without ever materializing the (N,d,d) tensor (huge for large N,d).
            for u in torch.unique(slot):
                gv = g[slot == u]                                        # (N_v,d)
                Hsum[u] += gv.t() @ gv
            cnt.index_add_(0, slot, torch.ones_like(slot, dtype=torch.float64))
        h.remove()
        self.model.zero_grad(set_to_none=True)
        # true mean-loss block: divide by the GLOBAL token count, so the block
        # magnitude keeps its N_v/N frequency scaling (same normalization as
        # neuron_blocks / last_layer_blocks). cnt is still returned so callers
        # can report how many distinct tokens were actually seen.
        Hsum /= max(1, n_tok)
        eigs = np.stack([torch.linalg.eigvalsh(Hsum[v]).cpu().numpy() for v in range(n_sel)])
        return eigs, cnt.cpu().numpy()

    # ---- embedding: EXACT Hessian block per token id (0-layer models only) ----
    def token_embedding_gn_blocks(self, emb_path="tok_emb", head_path="lm_head",
                                  max_tokens=None, token_ids=None, loss_type="ce"):
        """Exact per-token Hessian block of the embedding row e_v, valid ONLY for
        0-layer models (emb_out -> logits is z = W e_v + b, i.e. LINEAR in e_v, so
        the exact Hessian equals the Gauss-Newton block -- the residual term
        vanishes). W = lm_head.weight (C,d).

            H_v = (1/N) sum_{t: x_t=v} J_t^T (d^2 L / d z_t^2) J_t
                = (1/N) sum_{t: x_t=v} W^T S_t W                       (d x d)

          MSE-vs-one-hot:  d^2L/dz^2 = (2/C) I  (constant, position-independent),
            so H_v = (N_v/N) * (2/C) * W^T W  -- rank d, one shared matrix scaled
            by the token frequency N_v/N.
          CE:  S_t = diag(p_t) - p_t p_t^T with p_t = softmax(z_t); p_t depends on
            the position (via the added positional encoding), so we accumulate
            W^T S_t W over the positions where token v appears.

        Unlike token_embedding_blocks (empirical Fisher, g g^T), this is the TRUE
        curvature: for a frozen all-ones head the Fisher block collapses to rank 1
        while this GN block is full rank. Verified against torch.autograd Hessian
        to ~1e-15 (MSE) / ~1e-16 (CE)."""
        emb = self._module(emb_path)
        head = self._module(head_path)
        V_full = emb.num_embeddings
        d = emb.embedding_dim
        W = head.weight.detach().to(torch.float64)                    # (C,d)
        C_full = W.shape[0]
        if token_ids is not None:
            sel = torch.as_tensor(np.asarray(token_ids), dtype=torch.long,
                                  device=self.device)
        else:
            V = V_full if max_tokens is None else min(V_full, max_tokens)
            sel = torch.arange(V, device=self.device)
        n_sel = sel.numel()
        lut = torch.full((V_full,), -1, dtype=torch.long, device=self.device)
        lut[sel] = torch.arange(n_sel, device=self.device)
        cnt = torch.zeros(n_sel, dtype=torch.float64, device=self.device)
        n_tok = 0

        # MSE: curvature is constant (2/C) I, so H_v = (N_v/N)(2/C) W^T W. We only
        # need per-token counts -- no logits, one shared matrix.
        if loss_type == "mse":
            WtW = (W.t() @ W).to(self.device)                          # (d,d)
            store = {}
            def fhook(mod, inp, out):
                store["ids"] = inp[0].detach()
            h = emb.register_forward_hook(fhook)
            self.model.eval()
            with torch.no_grad():
                for _ in range(self.n_batches):
                    X, Y = self.get_batch()
                    self.model(X, Y)
                    ids = store["ids"].reshape(-1)
                    n_tok += ids.numel()
                    slot = lut[ids]; slot = slot[slot >= 0]
                    cnt.index_add_(0, slot, torch.ones_like(slot, dtype=torch.float64))
            h.remove()
            base = (2.0 / C_full) * WtW                                # shared (d,d)
            wbase = torch.linalg.eigvalsh(base).cpu().numpy()          # (d,)
            scale = (cnt / max(1, n_tok)).cpu().numpy()                # (n_sel,) = N_v/N
            eigs = scale[:, None] * wbase[None, :]                     # (n_sel,d)
            return eigs, cnt.cpu().numpy()

        # CE: S_t = diag(p_t) - p_t p_t^T depends on the position -> accumulate
        # W^T S_t W = (W^T diag(p) W) - (W^T p)(W^T p)^T per selected token.
        Hsum = torch.zeros((n_sel, d, d), dtype=torch.float64, device=self.device)
        store = {}
        def fhook(mod, inp, out):
            store["ids"] = inp[0].detach()
        h = emb.register_forward_hook(fhook)
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.n_batches):
                X, Y = self.get_batch()
                logits, _ = self.model(X, Y)
                ids = store["ids"].reshape(-1)                        # (N,)
                n_tok += ids.numel()
                P = F.softmax(logits.reshape(-1, logits.size(-1)), dim=-1).to(torch.float64)  # (N,C)
                slot = lut[ids]
                mask = slot >= 0
                slot = slot[mask]; P = P[mask]
                for u in torch.unique(slot):
                    Pu = P[slot == u]                                 # (N_v,C)
                    # sum_t W^T diag(p_t) W  =  W^T diag(sum_t p_t) W
                    WtDW = W.t() @ (Pu.sum(0).unsqueeze(1) * W)        # (d,d)
                    q = Pu @ W                                        # (N_v,d) = W^T p_t rows
                    Hsum[u] += WtDW - q.t() @ q
                cnt.index_add_(0, slot, torch.ones_like(slot, dtype=torch.float64))
        h.remove()
        Hsum /= max(1, n_tok)
        eigs = np.stack([torch.linalg.eigvalsh(Hsum[v]).cpu().numpy() for v in range(n_sel)])
        return eigs, cnt.cpu().numpy()

    # ---- hidden Linear: Fisher block per output neuron ----
    def neuron_blocks(self, path):
        lin = self._module(path)
        d_in = lin.in_features
        d_out = lin.out_features
        Hsum = torch.zeros((d_out, d_in, d_in), dtype=torch.float64, device=self.device)
        n_tok = 0

        store = {}
        def fhook(mod, inp, out):
            store["x"] = inp[0].detach()
            out.retain_grad()
            store["out"] = out
        h = lin.register_forward_hook(fhook)
        self.model.eval()
        for _ in range(self.n_batches):
            X, Y = self.get_batch()
            self.model.zero_grad(set_to_none=True)
            _, loss = self.model(X, Y)
            loss.backward()
            x = store["x"].reshape(-1, d_in).to(torch.float64)            # (N,d_in)
            g = store["out"].grad.reshape(-1, d_out).to(torch.float64)    # (N,d_out)
            N = x.shape[0]
            for i in range(d_out):
                s = g[:, i] ** 2                                          # (N,)
                Hsum[i] += x.t() @ (s.unsqueeze(1) * x)
            n_tok += N
        h.remove()
        self.model.zero_grad(set_to_none=True)
        Hsum /= max(1, n_tok)
        eigs = np.stack([torch.linalg.eigvalsh(Hsum[i]).cpu().numpy() for i in range(d_out)])
        return eigs

    # ---- attention Q/K: Fisher block per head ----
    def head_blocks(self, path, n_head, head_dim):
        """Per-head empirical-Fisher block of a Linear whose output is reshaped
        into (n_head, head_dim). Block h is (head_dim*d_in) x (head_dim*d_in),
        formed as U^T U / N with per-token rows u_t = x_t ⊗ g_{h,t}.

        We cache the (small) per-token activations x and per-head grads g across
        all batches, then build ONE (dim,dim) block at a time so peak memory is a
        single block, not (n_head, dim, dim) -- dim can be 6144."""
        lin = self._module(path)
        d_in = lin.in_features
        dim = head_dim * d_in

        store = {}
        def fhook(mod, inp, out):
            store["x"] = inp[0].detach()
            out.retain_grad()
            store["out"] = out
        h = lin.register_forward_hook(fhook)
        self.model.eval()

        xs, gs = [], []
        for _ in range(self.n_batches):
            X, Y = self.get_batch()
            self.model.zero_grad(set_to_none=True)
            _, loss = self.model(X, Y)
            loss.backward()
            xs.append(store["x"].reshape(-1, d_in).to(torch.float64))              # (N,d_in)
            gs.append(store["out"].grad.reshape(-1, n_head, head_dim).to(torch.float64))
        h.remove()
        self.model.zero_grad(set_to_none=True)
        x = torch.cat(xs, dim=0)                                                   # (Ntot,d_in)
        g = torch.cat(gs, dim=0)                                                   # (Ntot,H,hd)
        del xs, gs
        n_tok = x.shape[0]

        eigs = []
        for hh in range(n_head):
            gh = g[:, hh, :]                                                       # (Ntot,hd)
            U = torch.einsum("ni,nj->nij", x, gh).reshape(n_tok, dim)              # (Ntot,dim)
            H = (U.t() @ U) / max(1, n_tok)
            del U
            eigs.append(torch.linalg.eigvalsh(H).cpu().numpy())
            del H
        return np.stack(eigs)


# ----------------------------------------------------------------------------
# layer spec: display name -> how to block it
# ----------------------------------------------------------------------------
def default_layer_spec(n_head, head_dim, n_layer=1, block_type="transformer"):
    """Return an ordered list of (display_name, kind, kwargs) analysis items.

    embedding and lm_head always exist. The per-block sub-layers are only
    emitted for models that actually have transformer blocks (n_layer >= 1);
    with n_layer == 0 (embed + lm_head only) they are omitted, so the analyzer
    never tries to load or plot layers that don't exist.

    block_type selects how each block's FIRST sub-layer is blocked:
      * "transformer": attention -> Q/K per head, V/proj per neuron.
      * "mlp": the attention slot is a second FFN (see vanilla_model.Block), so
        it is blocked per output neuron like any Linear. The submodule attribute
        is still `attn`, hence the blocks.<i>.attn.c_fc / .c_proj paths below.
    """
    spec = [("embedding", "token", {"path": "tok_emb"})]
    for li in range(n_layer):
        p = f"blocks.{li}.attn"
        m = f"blocks.{li}.mlp"
        # display names keep the block index only when there is more than one
        # block, so the single-block case reads exactly as before.
        pre = "" if n_layer == 1 else f"b{li}_"
        if block_type == "mlp":
            spec += [
                (f"{pre}ffn1_fc",   "neuron", {"path": f"{p}.c_fc"}),
                (f"{pre}ffn1_proj", "neuron", {"path": f"{p}.c_proj"}),
                (f"{pre}ffn2_fc",   "neuron", {"path": f"{m}.c_fc"}),
                (f"{pre}ffn2_proj", "neuron", {"path": f"{m}.c_proj"}),
            ]
        else:
            spec += [
                (f"{pre}attn_wq",   "head",   {"path": f"{p}.wq"}),
                (f"{pre}attn_wk",   "head",   {"path": f"{p}.wk"}),
                (f"{pre}attn_wv",   "neuron", {"path": f"{p}.wv"}),
                (f"{pre}attn_proj", "neuron", {"path": f"{p}.wo"}),
                (f"{pre}mlp_fc",    "neuron", {"path": f"{m}.c_fc"}),
                (f"{pre}mlp_proj",  "neuron", {"path": f"{m}.c_proj"}),
            ]
    spec.append(("lm_head", "class", {"path": "lm_head"}))
    return spec


def compute_layer_eigs(nh, kind, kwargs, n_head, head_dim,
                       max_classes=256, max_tokens=256,
                       class_ids=None, token_ids=None, loss_type="ce",
                       emb_method="fisher"):
    """Dispatch to the right blocking routine, return (eigs, meta).

    emb_method selects how the embedding ("token") block is formed:
      * "fisher" -> empirical Fisher sum_t g_t g_t^T (works for any depth).
      * "gn"     -> EXACT Hessian = Gauss-Newton, ONLY valid for 0-layer models
                    (emb -> logits linear). Full rank; reveals the frequency
                    (N_v/N) scaling that the rank-deficient Fisher block hides.
    """
    if kind == "class":
        eigs = nh.last_layer_blocks(kwargs["path"], max_classes=max_classes,
                                    class_ids=class_ids, loss_type=loss_type)
        meta = {"unit": "token(class)", "loss_type": loss_type}
        if class_ids is not None:
            meta["token_select"] = "freq"
        return eigs, meta
    if kind == "token":
        if emb_method == "gn":
            eigs, cnt = nh.token_embedding_gn_blocks(
                kwargs["path"], max_tokens=max_tokens, token_ids=token_ids,
                loss_type=loss_type)
        else:
            eigs, cnt = nh.token_embedding_blocks(kwargs["path"], max_tokens=max_tokens,
                                                  token_ids=token_ids)
        meta = {"unit": "token", "n_seen": int((cnt > 0).sum()),
                "emb_method": emb_method, "_cnt": cnt}
        if token_ids is not None:
            meta["token_select"] = "freq"
        return eigs, meta
    if kind == "head":
        eigs = nh.head_blocks(kwargs["path"], n_head, head_dim)
        return eigs, {"unit": "head"}
    if kind == "neuron":
        eigs = nh.neuron_blocks(kwargs["path"])
        return eigs, {"unit": "neuron"}
    raise ValueError(f"unknown kind {kind}")


# ----------------------------------------------------------------------------
# top-level: analyze one layer of one checkpoint, save eigs + hetero matrices
# ----------------------------------------------------------------------------
def analyze_layer(nh, out_dir, tag, disp, kind, kwargs, n_head, head_dim,
                  max_classes=256, max_tokens=256, num_bins=64, device="cpu",
                  class_ids=None, token_ids=None, loss_type="ce",
                  emb_method="fisher"):
    save_dir = os.path.join(out_dir, tag)
    os.makedirs(save_dir, exist_ok=True)

    eigs, meta = compute_layer_eigs(nh, kind, kwargs, n_head, head_dim,
                                    max_classes=max_classes, max_tokens=max_tokens,
                                    class_ids=class_ids, token_ids=token_ids,
                                    loss_type=loss_type, emb_method=emb_method)
    np.save(os.path.join(save_dir, f"eigs_{disp}.npy"), eigs)
    # per-token occurrence counts (embedding only) -> for curvature-vs-freq plots
    cnt = meta.pop("_cnt", None)
    if cnt is not None:
        np.save(os.path.join(save_dir, f"counts_{disp}.npy"), cnt)

    edges = common_log_edges(eigs, num_bins)
    P = spectra_to_prob(eigs, edges)
    D_skl = pairwise_matrix(P, "skl", device=device)
    D_js = pairwise_matrix(P, "js", device=device)
    np.save(os.path.join(save_dir, f"hetero_{disp}_skl.npy"), D_skl)
    np.save(os.path.join(save_dir, f"hetero_{disp}_js.npy"), D_js)

    info = {"kind": kind, "unit": meta.get("unit"), "n_units": int(eigs.shape[0]),
            "skl_mean": hetero_mean(D_skl), "js_mean": hetero_mean(D_js)}
    info.update({k: v for k, v in meta.items() if k not in ("unit",)})
    with open(os.path.join(save_dir, f"summary_{disp}.json"), "w") as f:
        json.dump({"tag": tag, "layer": disp, **info}, f, indent=2)
    return info
