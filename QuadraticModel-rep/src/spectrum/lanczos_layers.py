"""
Lockstep 分片 Lanczos：K 个块的 Krylov 序列并排推进，共享每 minibatch 的前向/二阶图。

与 spectrum_ddp.lanczos_sharded 的关系：算法本体（Algorithm 1，全重正交 DGKS 两轮，
重正交在归一化前）逐行相同，两处推广：
  1. 作用的算子从全量 H 换成 H_bb（由 hvp_blocks 提供，投影是恒等改写）
  2. 同时维护 K 组 (Q, alpha, beta)，每步只调一次 matvec(blocks, vs)

分片：每块的坐标轴 [0, n_b) 按 world 切成 world 段，rank r 只存 Q_r(m × shard_r)。
注意这是**存储/并行**的切分，与参数张量边界无关（会切到张量中间去，无妨）。
重正交的系数 coeff = Σ_rank Q_local·w_local 由 all_reduce 得到，故各 rank 的
alpha/beta 完全一致。

★ Q 生命周期：一组块算完即 del + empty_cache，把显存交还**驱动**。
  torch.cuda.mem_get_info 问的是驱动的空闲量，只 del 不 empty_cache 的话
  内存仍留在 PyTorch 缓存池里，下一组会误判显存不足（实测导致 Q 回退 CPU、慢 2.5×）。
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import torch
import torch.distributed as dist


def block_seed(seed: int, name: str) -> int:
    """块相关但**跨进程稳定**的 seed。

    ⚠ 不能用内置 hash(name)：Python 对 str 的 hash 每进程随机（PYTHONHASHSEED），
    各 rank 会得到不同值 → 同一块的 v0 在各 rank 上不一致，而分片 Lanczos 是把各
    rank 的 shard 拼成一个向量的，结果会静默变成「不同随机向量的碎片拼接」，
    不报错但谱全错。crc32 是确定性的。
    """
    return (seed + 7919 * zlib.crc32(name.encode())) % (2 ** 31 - 1)


def shard_bounds(n, world):
    """把 [0, n) 尽量均匀切成 world 段（前 rem 段多 1）。与 spectrum_ddp 同式。"""
    base, rem = n // world, n % world
    out, off = [], 0
    for r in range(world):
        sz = base + (1 if r < rem else 0)
        out.append((off, off + sz))
        off += sz
    assert off == n
    return out


class LayerLanczosState:
    """单个块的分片 Lanczos 状态。"""

    def __init__(self, block, m, world, rank, device, q_device, seed, dtype=torch.float32):
        self.block, self.m = block, m
        self.bounds = shard_bounds(block.numel, world)
        self.s0, self.s1 = self.bounds[rank]
        self.shard = self.s1 - self.s0
        self.shard_sizes = [e - s for s, e in self.bounds]
        self.device, self.q_device, self.dtype = device, q_device, dtype

        # 初始向量：在 R^(n_b) 里各向同性（⚠ 不能 randn(N) 再截断 —— 那样范数不对，
        # 且谱密度的权重 E[<v0,u_i>²]=1/n_b 这个前提会破）。
        # 所有 rank 用同一 seed 生成完整 v0 再各取自己片 → 保证一致（见 block_seed）。
        # ⚠ 归一化必须在 float64 做：fp32 下对 8.4M 个元素求平方和，累加误差达 ~5e-4
        # （实测 ‖v0‖²=1.000461），会污染 Lanczos 的第一步。
        g = torch.Generator().manual_seed(block_seed(seed, block.name))
        v = torch.randn(block.numel, generator=g, dtype=torch.float64)
        v /= v.norm()
        # 跨 rank 一致性校验用的指纹：取完整 v0 的两个统计量（不是范数——
        # 各 rank 若各自生成不同的单位向量，分片范数平方和**仍然≈1**，范数检查会漏掉）
        self.fingerprint = (float(v.sum()), float((v * torch.arange(
            1, block.numel + 1, dtype=torch.float64)).sum()))
        self.Q = torch.zeros(m, self.shard, dtype=dtype, device=q_device)
        self.Q[0] = v[self.s0:self.s1].to(dtype).to(q_device)
        del v

        self.alpha = np.zeros(m, dtype=np.float64)
        self.beta = np.zeros(max(m - 1, 0), dtype=np.float64)
        self.done = False
        self.steps = 0

    def q_bytes(self):
        return self.m * self.shard * 4

    def gather_v(self):
        """all_gather 当前基向量 → 完整块向量（供 matvec）。"""
        j = self.steps
        local = self.Q[j].to(self.device)
        parts = [torch.empty(sz, dtype=self.dtype, device=self.device)
                 for sz in self.shard_sizes]
        dist.all_gather(parts, local)
        return torch.cat(parts)

    def absorb(self, w_full):
        """吃进 w = H_bb v_j，走一步三项递推 + 全重正交。"""
        j = self.steps
        w = w_full[self.s0:self.s1].to(self.q_device).clone()

        a = torch.dot(w, self.Q[j]).to(self.device)
        dist.all_reduce(a, op=dist.ReduceOp.SUM)
        self.alpha[j] = a.item()

        # 全重正交（DGKS 两轮），在归一化**之前**
        Qv = self.Q[: j + 1]
        for _ in range(2):
            coeff = torch.mv(Qv, w).to(self.device)
            dist.all_reduce(coeff, op=dist.ReduceOp.SUM)
            w = w - torch.mv(Qv.t(), coeff.to(self.q_device))

        b = torch.dot(w, w).to(self.device)
        dist.all_reduce(b, op=dist.ReduceOp.SUM)
        beta_j = math.sqrt(max(b.item(), 0.0))

        if beta_j < 1e-10 or j == self.m - 1:
            if j < self.m - 1:      # 提前收敛：截断
                self.alpha = self.alpha[: j + 1]
                self.beta = self.beta[:j]
                self.m = j + 1
            self.done = True
            self.steps = j + 1
            return beta_j

        self.beta[j] = beta_j
        self.Q[j + 1] = w / beta_j
        self.steps = j + 1
        return beta_j

    def tridiag(self):
        return self.alpha, self.beta

    def free(self):
        self.Q = None


def lockstep_lanczos(matvec, blocks, m_of, world, rank, device, store_device,
                     seed, reserve_bytes, log=print, log_every=25):
    """K 个块并排跑 Lanczos。matvec(blocks, vs) -> [H_bb v_b]。

    返回 {block_name: (alpha, beta)}。
    """
    # Q 放哪：先纯算术估总字节数（不分配任何东西），再决定 device。
    # ⚠ 别用「先建一批 state 探测」的写法：那会为每块真的生成一遍 n_b 的 float64
    # 随机向量（embd/head 各 8.4M、层块 12.6M），纯属浪费。
    total = sum(m_of(b) * (shard_bounds(b.numel, world)[rank][1]
                           - shard_bounds(b.numel, world)[rank][0]) * 4
                for b in blocks)
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        if total + reserve_bytes <= free:
            q_device = device
        else:
            q_device = store_device
            log(f"⚠ 显存不足放 Q({total/1e9:.1f}GB)+HVP保留({reserve_bytes/1e9:.0f}GB)"
                f">空闲({free/1e9:.1f}GB)，Q 回退 CPU")
    else:
        q_device = store_device

    states = [LayerLanczosState(b, m_of(b), world, rank, device, q_device, seed)
              for b in blocks]

    # 跨 rank 一致性自检：比对 v0 的**指纹**（全向量的两个线性统计量），而不是范数。
    # 范数检查不管用：各 rank 若各自生成不同的单位向量，分片范数平方和仍≈1。
    # 指纹是各 rank 独立算出的完整 v0 的统计量，只要 seed 一致就必然逐位相同。
    if dist.is_initialized() and world > 1:
        for s in states:
            fp = torch.tensor(s.fingerprint, dtype=torch.float64, device=device)
            ref = fp.clone()
            dist.broadcast(ref, src=0)
            d = float((fp - ref).abs().max())
            assert d == 0.0, (
                f"块 {s.block.name} 的 v0 跨 rank 不一致（指纹差 {d:.3e}）"
                f"——各 rank 生成了不同的随机向量")

    log(f"  Q 放置: {q_device}  总计 {total/1e9:.2f}GB/rank  ({len(states)} 块 lockstep)")
    for s in states:
        log(f"    {s.block.name:18s} n_b={s.block.numel:>11,}  m={s.m}  "
            f"Q={s.q_bytes()/1e9:.2f}GB/rank")

    step = 0
    while True:
        active = [s for s in states if not s.done]
        if not active:
            break
        vs = [s.gather_v() for s in active]
        ws = matvec([s.block for s in active], vs)     # ← 一次 prepare 服务全部 active
        for s, w in zip(active, ws):
            s.absorb(w)
        step += 1
        if step % log_every == 0:
            log(f"  lockstep step {step}  active={len(active)}  "
                + " ".join(f"{s.block.name}:{s.steps}/{s.m}" for s in active[:4])
                + (" ..." if len(active) > 4 else ""))

    out = {s.block.name: s.tridiag() for s in states}
    # ★ 释放 Q 并把显存还给驱动
    for s in states:
        s.free()
    if q_device.type == "cuda":
        torch.cuda.empty_cache()
    return out
