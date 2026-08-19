"""
块 exact Hessian/GN **矩阵本体**（不是谱）：对给定块打单位向量，逐行精确建出
稠密 H_bb 并落盘 npz，供 scripts/plot_exact_hessian.py 画结构热图。

与 dense_block_eig 的区别：那边建完直接 eigvalsh 只存特征值；这边要的是矩阵
本身（画"Hessian 长什么样"），且用 SharedBatchOp.apply_batched（is_grads_batched
批量 VJP）加速 —— n_b=8192 时逐行 apply 太慢。

块坐标 = layers.neuron_subblock 的 neuron-major：idx = k*d_in + d。热图对角上
第 k 个 d_in×d_in 块 = 第 k 个选定输出神经元自己的块，非对角 = 跨神经元耦合。

热路径用 **零值叶子注入**（dense_block_matrix_injected）：往前向里注入一个
(n_sel, d_in) 的零叶子 delta，scatter 加到目标层参数上，再对 delta 求二阶导。
因为 `u = a @ W` 且输入激活 a 不依赖 W，这与直接对 W 求导逐元素相等（恒等式）。
好处：cotangent 从「与整个 stacked 张量同形」(mlp_up 是 5033 万元素 = 192 MiB
fp32、非零密度 2e-8) 降到 n_b 个 float，chunk 才开得大。旧的
dense_block_matrix（inputs=整个叶子）保留作对拍参照与回退。

用法（SCO 8 卡）:
  torchrun --standalone --nproc_per_node=8 src/analysis/exact_block_hessian.py \
      --ckpt checkpoints_b64/ckpt_p0.pt \
      --layer 1 --param mlp_up --neurons top:8 --kind hessian \
      --n_tokens 65536 --per 8 --chunk 256 \
      --out outputs/exact_hessian/p0_layer01_mlp_up_hessian.npz

本地正确性验证（tiny 模型 vs 暴力全 Hessian，CPU 秒级）:
  python src/analysis/exact_block_hessian.py --verify
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.model.layers import neuron_subblock                            # noqa: E402
from src.model.model import apply_rope, rmsnorm                         # noqa: E402
from src.spectrum.hvp import _MATH_SDPA                                 # noqa: E402
from src.spectrum.hvp_layers import SharedBatchOp                       # noqa: E402
from src.spectrum.spectrum_ddp import load_checkpoint, make_local_batches  # noqa: E402

# 支持注入的参数 → (输入激活维 d_in, 输出神经元维 d_out)
_INJECT_DIMS = {
    "mlp_up":    lambda c: (c.D, c.M),
    "mlp_head":  lambda c: (c.M, c.D),
    "attn_v":    lambda c: (c.D, c.H * c.K),
    "attn_head": lambda c: (c.H * c.K, c.D),
}


def _forward_injected(model, x, l, pname, ids, delta):
    """model.forward 的复刻，但第 l 层 pname 的选定输出神经元列上加了 delta
    （(n_sel, d_in) 的**零值**叶子）：W_eff = W[l] + scatter(deltaᵀ)。

    因为 W 只经 `out = a @ W` 进入网络、且输入激活 a 不依赖 W，所以
    ∂²loss/∂delta² 与 ∂²loss/∂W 在这些坐标上逐元素相等（恒等式，非近似）。
    delta 是 (n_sel, d_in) 的小张量 → cotangent 也只有这么大，这是相对
    inputs=整个 stacked 叶子的全部加速来源。
    返回 logits。"""
    cfg = model.cfg
    P = getattr(model, pname)
    d_in, d_out = _INJECT_DIMS[pname](cfg)
    W_eff = P[l].reshape(d_in, d_out).index_add(1, ids, delta.t()).reshape(P[l].shape)

    def W(pn, li):
        """第 li 层 pn 的权重；命中注入目标则用 W_eff。"""
        return W_eff if (li == l and pn == pname) else getattr(model, pn)[li]

    h = model.embd[x]
    for li in range(cfg.L):
        hn = rmsnorm(h, cfg.norm_eps)
        q = torch.einsum("...D,DHK->...HK", hn, W("attn_q", li))
        k = torch.einsum("...D,DHK->...HK", hn, W("attn_k", li))
        v = torch.einsum("...D,DHK->...HK", hn, W("attn_v", li))
        q = apply_rope(rmsnorm(q, cfg.norm_eps), cfg.rope_freq)
        k = apply_rope(rmsnorm(k, cfg.norm_eps), cfg.rope_freq)
        qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
        a = a.transpose(1, 2)
        h = h + torch.einsum("...HK,HKD->...D", a, W("attn_head", li)) / cfg.L
        hn = rmsnorm(h, cfg.norm_eps)
        u = F.gelu(torch.einsum("...D,DM->...M", hn, W("mlp_up", li)),
                   approximate="tanh")
        h = h + torch.einsum("...M,MD->...D", u, W("mlp_head", li)) / cfg.L
    h = rmsnorm(h, cfg.norm_eps)
    return torch.einsum("...TD,DV->...TV", h, model.head)


class InjectedBatchOp:
    """SharedBatchOp 的注入版：对零值小叶子 delta 求二阶导，而非对整个 stacked
    叶子。接口同 SharedBatchOp（prepare/apply_batched/release），但块固定为
    构造时给的 (layer, pname, neuron_ids)，且不支持预条件。"""

    def __init__(self, model, x, y, kind, l, pname, ids, n_sel, d_in):
        self.model, self.x, self.y, self.kind = model, x, y, kind
        self.l, self.pname, self.ids = l, pname, ids
        self.n_sel, self.d_in = n_sel, d_in
        self._ready = False

    def prepare(self):
        pdtype = next(self.model.parameters()).dtype
        self.delta = torch.zeros(self.n_sel, self.d_in, dtype=pdtype,
                                 device=self.x.device, requires_grad=True)
        with sdpa_kernel(_MATH_SDPA):     # flash/efficient/cudnn 无二阶反向
            logits = _forward_injected(self.model, self.x, self.l, self.pname,
                                       self.ids, self.delta)
        self.logits = logits
        if self.kind == "gn":
            self.q = F.softmax(logits, dim=-1)
            self.u = torch.zeros_like(logits, requires_grad=True)
            (self.gu,) = torch.autograd.grad(
                logits, self.delta, grad_outputs=self.u,
                create_graph=True, retain_graph=True)
        else:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), self.y.reshape(-1),
                ignore_index=-1)
            (self.grads,) = torch.autograd.grad(
                loss, self.delta, create_graph=True, retain_graph=True)
        self._ready = True

    def apply_batched(self, V):
        """V (K, n_b) → (K, n_b)，一次 autograd 调用算 K 行 H_bb。"""
        assert self._ready, "先调用 prepare()"
        K = V.shape[0]
        Vf = V.reshape(K, self.n_sel, self.d_in)
        if self.kind == "gn":
            (jvp,) = torch.autograd.grad(
                self.gu, self.u, grad_outputs=Vf,
                retain_graph=True, is_grads_batched=True)      # (K,B,T,V)
            jvp = jvp - (self.q.unsqueeze(0) * jvp).sum(-1, keepdim=True)
            cot = self.q.unsqueeze(0) * jvp / self.y.numel()
            (rows,) = torch.autograd.grad(
                self.logits, self.delta, grad_outputs=cot,
                retain_graph=True, is_grads_batched=True)
        else:
            (rows,) = torch.autograd.grad(
                self.grads, self.delta, grad_outputs=Vf,
                retain_graph=True, is_grads_batched=True)
        return rows.reshape(K, -1)

    def release(self):
        for a in ("logits", "grads", "gu", "u", "q", "delta"):
            if hasattr(self, a):
                delattr(self, a)
        self._ready = False


def dense_block_matrix_injected(model, batches, nb_global, l, pname, neuron_ids,
                                kind, device, chunk=32, log=print):
    """注入版稠密 H_bb（fp64）。每 minibatch 一次 prepare，之后 n_b 行按 chunk
    批量。显存瓶颈是 per-lane 的 logits 尺寸张量 (K,B,T,V)，即 chunk×per 的乘积，
    OOM 时 chunk 减半重试。返回 (H_local_sum, n_b)：**未除 nb_global**，
    留给调用方 all_reduce 后统一归一化。"""
    ids = torch.as_tensor(list(neuron_ids), dtype=torch.long, device=device)
    d_in, _ = _INJECT_DIMS[pname](model.cfg)
    n_sel = len(neuron_ids)
    n = n_sel * d_in
    H = torch.zeros(n, n, dtype=torch.float64, device=device)
    pdtype = next(model.parameters()).dtype
    eye = torch.eye(n, dtype=pdtype, device=device)
    for bi, (x, y) in enumerate(batches):
        op = InjectedBatchOp(model, x, y, kind, l, pname, ids, n_sel, d_in)
        op.prepare()
        t0, i = time.time(), 0
        while i < n:
            j = min(i + chunk, n)
            try:
                rows = op.apply_batched(eye[i:j])
            except torch.OutOfMemoryError:
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
                log(f"⚠ OOM，chunk 减半 → {chunk}")
                torch.cuda.empty_cache()
                continue
            H[i:j] += rows.double()
            i = j
        op.release()
        del op
        log(f"  batch {bi+1}/{len(batches)}  {n} 行  chunk={chunk}  "
            f"{time.time()-t0:.1f}s")
    return H, n


def dense_block_matrix(model, batches, nb_global, block, kind, device,
                       chunk=8, use_batched=True, log=print):
    """精确建稠密 H_bb (n_b×n_b, fp64)。每 minibatch 一次 prepare（贵），
    之后 n_b 次 VJP 按 chunk 批量（is_grads_batched）；OOM 时 chunk 减半重试
    （批量 vmap 每条 lane 都持整个双反向图，显存 ∝ chunk），减到 1 即逐行。
    vmap 与 MATH-sdpa 二阶反向若不兼容（非 OOM 的 RuntimeError）则回退逐行。
    返回 (H, 对称残差)。"""
    n = block.numel
    H = torch.zeros(n, n, dtype=torch.float64)
    pdtype = next(model.parameters()).dtype
    eye = torch.eye(n, dtype=pdtype, device=device)
    for bi, (x, y) in enumerate(batches):
        op = SharedBatchOp(model, x, y, kind, precond=None)
        op.prepare([block])
        t0 = time.time()
        i = 0
        while i < n:
            j = min(i + chunk, n)
            if use_batched:
                try:
                    rows = op.apply_batched(block, eye[i:j])
                except torch.cuda.OutOfMemoryError:
                    if chunk > 1:
                        chunk = max(1, chunk // 2)
                        log(f"⚠ OOM，chunk 减半 → {chunk}")
                        torch.cuda.empty_cache()
                        continue
                    log("⚠ chunk=1 仍 OOM，回退逐行 apply")
                    use_batched = False
                    torch.cuda.empty_cache()
                    continue
                except RuntimeError as e:
                    log(f"⚠ apply_batched 失败（{e}），回退逐行")
                    use_batched = False
                    continue
            else:
                rows = torch.stack([op.apply(block, eye[k]) for k in range(i, j)])
            H[i:j] += rows.double().cpu()
            i = j
        op.release()
        log(f"  batch {bi+1}/{len(batches)}  {n} 行  chunk={chunk}  "
            f"{time.time()-t0:.1f}s")
    H /= nb_global
    sym_resid = float((H - H.T).abs().max())
    return 0.5 * (H + H.T), sym_resid


def pick_neurons(model, batches, layer, param, spec, device, log=print):
    """解析 --neurons：'top:K'（按 GN 对角 Σ_t g_i²·‖a‖² 选曲率最大的 K 个输出
    神经元）| 'random:K' | 'first:K' | 逗号分隔显式 id。"""
    if "," in spec or spec.isdigit():
        return [int(t) for t in spec.split(",")]
    mode, k = spec.split(":")
    k = int(k)
    from src.model.blocks import UnitHessian, unit_layout
    n_units = unit_layout(model.cfg)[param]["n_units"]
    if mode == "first":
        return list(range(k))
    if mode == "random":
        g = torch.Generator().manual_seed(0)
        return torch.randperm(n_units, generator=g)[:k].sort().values.tolist()
    if mode != "top":
        raise SystemExit(f"未知 --neurons 模式 {mode}")
    # top：借 UnitHessian 的内联 forward 捕获 (a, g)，打分 = Σ_t g_i²·‖a_t‖²
    # （= 该神经元 GN 块的 trace）。只用一阶反传，秒级。
    want, out_key, in_key = {
        "attn_v":    ("attn_qkv", "v_raw",    "attn_in"),
        "attn_head": ("attn_o",   "attn_out", "attn_agg"),
        "mlp_up":    ("mlp_up",   "u_raw",    "mlp_in"),
        "mlp_head":  ("mlp_head", "mlp_out",  "gelu_u"),
    }[param]
    it = iter(batches)
    uh = UnitHessian(model, lambda: next(it), n_batches=1, device=device)
    score = torch.zeros(n_units, dtype=torch.float64, device=device)
    with torch.enable_grad():
        X, Y = uh.get_batch()
        model.zero_grad(set_to_none=True)
        loss, cap = uh._forward(X, Y, layer, {want})
        loss.backward()
        a = cap[in_key].reshape(-1, cap[in_key].shape[-1]).double()
        g = cap[out_key].grad.reshape(-1, n_units).double()
        score += (g ** 2).T @ (a ** 2).sum(1)
        model.zero_grad(set_to_none=True)
    ids = torch.argsort(score, descending=True)[:k].sort().values.tolist()
    log(f"  top-{k} 神经元（按 GN trace）: {ids}")
    return ids


# --------------------------------------------------------------------------
# --verify：tiny 模型 vs 暴力全 Hessian（复用 verify_layers 的 brute_*）
# --------------------------------------------------------------------------
def run_verify():
    from src.model.model import Transformer, TransformerConfig
    from src.analysis.verify_layers import brute_hessian, brute_gn

    torch.manual_seed(0)
    cfg = TransformerConfig(D=8, L=2, M=8, H=2, K=4, V=16, seq_len=4)
    model = Transformer(cfg).double()
    # ⚠ 必须打破 head 的零初始化（reset_parameters 里 head.zero_()）：head=0 →
    # logits≡0 → mlp_up 的 Hessian 块恒为零矩阵，对拍变成 0==0，测不出任何 bug
    # （见 verify_layers.py:88 的同一教训）。verify 验证的是算法实现在任意参数点
    # 与暴力法一致，参数取随机点完全合法。
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.5)
    x = torch.randint(0, cfg.V, (2, cfg.seq_len))
    y = torch.randint(0, cfg.V, (2, cfg.seq_len))

    block = neuron_subblock(model, l=1, pname="mlp_up", neuron_ids=[1, 3, 6])
    n = block.numel
    assert n == 3 * cfg.D

    # IndexBlock 坐标 → 全局 flat 索引（与暴力矩阵对齐）
    off, base = {}, 0
    for nm, p in model.named_parameters():
        off[nm] = base
        base += p.numel()
    gidx = torch.cat([off[pn] + ix for pn, ix in block.indices])

    for kind, brute in (("hessian", brute_hessian), ("gn", brute_gn)):
        Hfull = brute(model, x, y).double()
        ref = Hfull[gidx][:, gidx]
        scale = ref.abs().max().item()
        assert scale > 0, f"{kind} 参考矩阵全零 —— 空对拍，verify 无效"
        # rmsnorm/apply_rope 内部硬转 .float()，实际精度只有 fp32：
        # 阈值取暴力矩阵自身不对称度的 100 倍与 1e-6 的 max（同 verify_layers）
        asym = float((Hfull - Hfull.T).abs().max())
        tol = max(asym * 100, 1e-6)
        H, sym = dense_block_matrix(
            model, [(x, y)], nb_global=1, block=block, kind=kind,
            device=torch.device("cpu"), chunk=5, log=lambda *a: None)
        err = (H - ref).abs().max().item()
        print(f"[verify {kind:7s}] max|Δ|={err:.3e}  (ref |max|={scale:.3e}, "
              f"tol={tol:.1e})  对称残差={sym:.3e}")
        assert err < tol, f"{kind} 对拍失败"
        # 逐行路径与批量路径一致性
        H2, _ = dense_block_matrix(
            model, [(x, y)], nb_global=1, block=block, kind=kind,
            device=torch.device("cpu"), chunk=5, use_batched=False,
            log=lambda *a: None)
        err2 = (H - H2).abs().max().item()
        print(f"[verify {kind:7s}] batched vs 逐行 max|Δ|={err2:.3e}")
        assert err2 < 1e-10
        # 注入路径（生产热路径）vs 同一暴力参照：验证「对 delta 求导 == 对 W 求导」
        Hinj, n_inj = dense_block_matrix_injected(
            model, [(x, y)], nb_global=1, l=1, pname="mlp_up",
            neuron_ids=[1, 3, 6], kind=kind, device=torch.device("cpu"),
            chunk=5, log=lambda *a: None)
        assert n_inj == n
        Hinj = 0.5 * (Hinj + Hinj.T)
        err3 = (Hinj - ref).abs().max().item()
        print(f"[verify {kind:7s}] 注入 vs 暴力 max|Δ|={err3:.3e}  (tol={tol:.1e})")
        assert err3 < tol, f"{kind} 注入路径对拍失败"
    print("✅ verify 通过")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="tiny 模型暴力对拍后退出")
    ap.add_argument("--ckpt")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--param", default="mlp_up")
    ap.add_argument("--neurons", default="top:8",
                    help="top:K | random:K | first:K | 逗号分隔显式 id")
    ap.add_argument("--kind", default="hessian", choices=["hessian", "gn"])
    ap.add_argument("--n_tokens", type=int, default=65536)
    ap.add_argument("--per", type=int, default=8,
                    help="每 minibatch 序列数（前向图 ∝ per；调大才吃满 GPU）")
    ap.add_argument("--chunk", type=int, default=256,
                    help="is_grads_batched 每批行数（显存 ∝ chunk×per，OOM 自动减半）")
    ap.add_argument("--ema", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", default="nccl")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.verify:
        run_verify()
        return
    if not args.ckpt or not args.out:
        raise SystemExit("--ckpt 与 --out 必填（或用 --verify）")

    # torchrun 下走 DDP（各 rank 分自己那几个 minibatch，最后 all_reduce）；
    # 裸 python 则单卡。⚠ 申请多卡却裸 python 跑 = 只用 1 卡（利用率恒 1/N）。
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group(backend=args.backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        rank, world = 0, 1
        device = (torch.device("cuda", 0) if torch.cuda.is_available()
                  else torch.device("cpu"))

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    log(f"world={world} device={device}  kind={args.kind}  "
        f"layer={args.layer} param={args.param}")
    if args.param not in _INJECT_DIMS:
        raise SystemExit(f"--param {args.param} 不支持注入"
                         f"（可用 {list(_INJECT_DIMS)}）")
    model, cfg, optim_name, _precond, _precond_raw = load_checkpoint(
        args.ckpt, args.ema, device)

    batches, nb_global = make_local_batches(
        cfg, args.n_tokens, world, rank, device, args.seed, per=args.per)
    log(f"local batches={len(batches)} nb_global={nb_global} "
        f"tokens={nb_global*args.per*cfg.seq_len:,}")

    # 神经元选取必须全 rank 一致：rank0 选好后广播（各 rank 的 batch 不同，
    # 独立按 GN trace 打分会选出不同神经元 → 拼出来的 H 是错的）
    neuron_ids = pick_neurons(model, batches, args.layer, args.param,
                              args.neurons, device, log=log)
    if ddp:
        t = torch.tensor(neuron_ids, dtype=torch.long, device=device)
        dist.broadcast(t, src=0)
        neuron_ids = t.tolist()
    d_in, _ = _INJECT_DIMS[args.param](cfg)
    n_b = len(neuron_ids) * d_in
    log(f"neurons={neuron_ids}  d_in={d_in}  H 为 {n_b}² fp64 "
        f"≈{n_b**2*8/2**30:.2f} GiB")

    t0 = time.time()
    H, _ = dense_block_matrix_injected(
        model, batches, nb_global, args.layer, args.param, neuron_ids,
        args.kind, device, chunk=args.chunk, log=log)
    if ddp:
        dist.all_reduce(H, op=dist.ReduceOp.SUM)   # Σ(每 minibatch 均值)
    H = (H / nb_global).cpu()
    sym_resid = float((H - H.T).abs().max())
    H = 0.5 * (H + H.T)
    log(f"总耗时 {time.time()-t0:.0f}s  对称残差={sym_resid:.3e}")

    if rank != 0:
        dist.destroy_process_group()
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta = {"ckpt": args.ckpt, "layer": args.layer, "param": args.param,
            "kind": args.kind, "neuron_ids": neuron_ids, "d_in": d_in,
            "n_tokens": nb_global * args.per * cfg.seq_len,
            "ema": args.ema, "seed": args.seed, "optim_name": optim_name,
            "precond": "none", "sym_resid": sym_resid}
    np.savez(args.out, H=H.numpy(),
             neuron_ids=np.array(neuron_ids), d_in=d_in,
             meta=json.dumps(meta, ensure_ascii=False))
    print(f"→ {args.out}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
