"""
块 Hessian / 块 GN 的 matvec，支持 **lockstep 共享前向图**。

数学：block b 的算子是 H_bb = P_bᵀ H P_b（P_b = 零填充嵌入）。实现上两次投影都不用
显式的零填充/切片：
  - P_b v_b（右投影）：只在块 b 的坐标上填 v_b，其余为 0 → 内积 <grad, v> 里只有块 b 非零
  - P_bᵀ(⋯)（左投影）：外层 autograd.grad 的 inputs= 限制到块所涉张量，再切出块内那段
两者都是恒等改写，不是近似。

为什么能 lockstep（省的来源）——按对 v 的依赖性把算子因式分解：

  GN:      G = (1/T) Σ_t J_tᵀ A_t J_t,  A_t = diag(q_t) − q_t q_tᵀ
    v-无关: forward → logits → q;  gu = ∂<u,logits>/∂θ  （create_graph，建二阶图，最贵）
    v-依赖: Jv = ∂<gu,v>/∂u;  cot = A(Jv)/T;  Jᵀcot          （一次 VJP）

  Hessian:
    v-无关: forward → loss;  grads = ∂loss/∂θ（create_graph，最贵）
    v-依赖: ∂<grads,v>/∂θ                                     （一次 VJP）

最贵的那一步恰好落在 v-无关一侧 → 同一个 minibatch 的前向/二阶图可以喂给 K 个块的
K 个不同 v。降低的是**每次 matvec 的成本**，不是 matvec 的次数：K 个块必须 K 条独立的
Krylov 序列（因为 P_bᵀ H (P_a v_a) = H_ba v_a ≠ 0，交叉块是确定性污染，采样平均消不掉）。

成本（S=v-无关部分, c=v-依赖部分，K 条序列各 m 步）：
    分开跑  Σ_k m_k (S + c)      一起排  max_k m_k · S + Σ_k m_k · c
S 只按步数付、不按序列数付。

⚠ inputs= 截断反向路径的收益仅对**整张量块**（embd/head）成立。本模型参数跨层 stack
（attn_q 是单个 (L,D,H,K) 叶子），故 per-layer 块的 inputs= 仍是整个 stacked 张量，
反向路径长度不变，只省了投影本身。见 blocks.py 顶部说明。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

_MATH_SDPA = [SDPBackend.MATH]


def _zero_filled(block, v_block, pmap):
    """把块向量 v_block 写进「与各参数同形、块外为 0」的 dict（= P_b v_b）。"""
    if hasattr(block, "fill"):          # IndexBlock：散点坐标，向量化写入
        return block.fill(v_block, pmap)
    out = {}
    for pn in block.params:
        out[pn] = torch.zeros_like(pmap[pn])
    for seg, s in zip(block.split(v_block), block.specs):
        out[s.param].reshape(-1)[s.start:s.stop] = seg
    return out


def _gather_block(block, by_name, out=None):
    """从 {param_name: 同形张量} 里抽出块 b 的各段并拼成块向量（= P_bᵀ ⋯）。"""
    segs = block.slices_of(by_name)
    return torch.cat(segs) if out is None else torch.cat(segs, out=out)


def _precond_seg(precond, s):
    """取预条件器在 spec s 上的那一段。

    ⚠ 预条件器有两种形状（见 spectrum_ddp.load_checkpoint）：
      - adam: 与参数同形的张量 → 要按 [start:stop] 切片
      - raw : **0 维标量**（CompleteP 每层 lr 乘子 √(pre·post)，全张量共用一个数）
              → 不能切片，直接广播。切了会得到空张量（首版 bug：
              size of tensor a (1048576) must match tensor b (0)）。
    原版 hvp.apply_preconditioner 是整张量逐元素乘，标量自动广播，故没暴露这点。
    """
    p = precond[s.param]
    if p.dim() == 0 or p.numel() == 1:
        return p
    return p.reshape(-1)[s.start:s.stop]


def _precond_block(block, v_block, precond):
    """对块向量逐元素乘预条件器的对应段。对角矩阵与坐标投影可交换，
    故 P_bᵀ(P H P)P_b = (PHP)_bb —— 语义正是「预条件后算子的块」。"""
    if precond is None:
        return v_block
    segs = []
    for seg, s in zip(block.split(v_block), block.specs):
        segs.append(seg * _precond_seg(precond, s))
    return torch.cat(segs)


class SharedBatchOp:
    """单个 minibatch 上的块 matvec；v-无关部分只算一次，可服务多个块/多个 v。

    用法：
        op = SharedBatchOp(model, x, y, kind, precond)
        op.prepare(blocks)                 # forward + create_graph 反向（最贵，一次）
        for b, v in zip(blocks, vs):
            acc[b] += op.apply(b, v)       # 每块一次便宜的 VJP
        op.release()
    """

    def __init__(self, model, x, y, kind, precond):
        self.model, self.x, self.y = model, x, y
        self.kind, self.precond = kind, precond
        self.pmap = dict(model.named_parameters())
        self._ready = False

    def prepare(self, blocks):
        # 本组块涉及的叶子张量并集：只对这些建二阶图
        names = []
        for b in blocks:
            for pn in b.params:
                if pn not in names:
                    names.append(pn)
        self.names = names
        self.inputs = [self.pmap[n] for n in names]

        with sdpa_kernel(_MATH_SDPA):
            logits, _ = self.model(self.x, self.y)
        self.logits = logits

        if self.kind == "gn":
            self.q = F.softmax(logits, dim=-1)
            self.u = torch.zeros_like(logits, requires_grad=True)
            # gu = ∂<u, logits>/∂θ —— 只依赖 u，与 v 无关 → 全组共享
            self.gu = torch.autograd.grad(
                outputs=logits, inputs=self.inputs, grad_outputs=self.u,
                create_graph=True, retain_graph=True,
            )
            self.gu_map = dict(zip(names, self.gu))
        else:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), self.y.reshape(-1),
                ignore_index=-1,
            )
            # grads = ∂loss/∂θ —— 与 v 无关 → 全组共享
            self.grads = torch.autograd.grad(
                outputs=loss, inputs=self.inputs,
                create_graph=True, retain_graph=True,
            )
            self.grads_map = dict(zip(names, self.grads))
        self._ready = True

    def apply(self, block, v_block):
        """返回 H_bb v_b（块坐标，(n_b,)）。含预条件则为 (P H P)_bb v_b。"""
        assert self._ready, "先调用 prepare()"
        vp = _precond_block(block, v_block, self.precond)      # 右预条件
        v_filled = _zero_filled(block, vp, self.pmap)          # P_b v_b

        binputs = [self.pmap[pn] for pn in block.params]

        if self.kind == "gn":
            inner = self.logits.new_zeros(())
            for pn in block.params:
                inner = inner + (self.gu_map[pn] * v_filled[pn]).sum()
            (jvp,) = torch.autograd.grad(inner, self.u, retain_graph=True)   # J v
            # 中心化：去掉 softmax 零空间分量，等价于乘 A_t = diag(q)−qqᵀ
            jvp = jvp - (self.q * jvp).sum(-1, keepdim=True)
            cot = self.q * jvp / self.y.numel()
            gv = torch.autograd.grad(
                outputs=self.logits, inputs=binputs,
                grad_outputs=cot, retain_graph=True,
            )
        else:
            gv_scalar = self.logits.new_zeros(())
            for pn in block.params:
                gv_scalar = gv_scalar + (self.grads_map[pn] * v_filled[pn]).sum()
            gv = torch.autograd.grad(gv_scalar, binputs, retain_graph=True)

        out = _gather_block(block, dict(zip(block.params, gv)))   # P_bᵀ ⋯
        return _precond_block(block, out, self.precond)           # 左预条件

    def apply_batched(self, block, V):
        """批量版 apply：V (K, n_b) → (K, n_b)，一次 autograd 调用算 K 行。

        技巧：<grads, v> 对 θ 的导数 = autograd.grad(outputs=grads, inputs=θ,
        grad_outputs=v)，故批量只需把 grad_outputs 堆上 K 维 + is_grads_batched=True
        （内部走 vmap）。GN 的 J·v 一步同理（outputs=gu, inputs=u）。
        ⚠ 仅支持 precond=None（IndexBlock 无 specs，预条件切段不适用）。
        """
        assert self._ready, "先调用 prepare()"
        assert self.precond is None, "apply_batched 仅支持无预条件"
        K = V.shape[0]
        pmap = self.pmap
        # P_b v_k：每参数堆成 (K, *shape)
        Vf = {pn: torch.zeros((K,) + pmap[pn].shape,
                              dtype=V.dtype, device=V.device)
              for pn in block.params}
        for k in range(K):
            filled = _zero_filled(block, V[k], pmap)
            for pn in block.params:
                Vf[pn][k] = filled[pn]

        binputs = [pmap[pn] for pn in block.params]

        if self.kind == "gn":
            jvp = torch.autograd.grad(
                outputs=[self.gu_map[pn] for pn in block.params],
                inputs=self.u,
                grad_outputs=[Vf[pn] for pn in block.params],
                retain_graph=True, is_grads_batched=True,
            )[0]                                                   # (K,B,T,V)
            jvp = jvp - (self.q.unsqueeze(0) * jvp).sum(-1, keepdim=True)
            cot = self.q.unsqueeze(0) * jvp / self.y.numel()
            gv = torch.autograd.grad(
                outputs=self.logits, inputs=binputs, grad_outputs=cot,
                retain_graph=True, is_grads_batched=True,
            )                                                      # 各 (K,*shape)
        else:
            gv = torch.autograd.grad(
                outputs=[self.grads_map[pn] for pn in block.params],
                inputs=binputs,
                grad_outputs=[Vf[pn] for pn in block.params],
                retain_graph=True, is_grads_batched=True,
            )                                                      # 各 (K,*shape)

        gv_map = {pn: g for pn, g in zip(block.params, gv)}
        rows = []
        for k in range(K):
            rows.append(_gather_block(block, {pn: gv_map[pn][k] for pn in block.params}))
        return torch.stack(rows)

    def release(self):
        for attr in ("logits", "q", "u", "gu", "gu_map", "grads", "grads_map", "inputs"):
            if hasattr(self, attr):
                delattr(self, attr)
        self._ready = False


def make_layer_matvec(model, batches, nb_global, kind, precond, dist=None):
    """返回 matvec(blocks, vs) -> list[H_bb v_b]，在本地 batches 上求和后跨 rank 归约。

    归一化与 spectrum_ddp.make_dist_hvp 完全一致：每 minibatch 的 HVP 已是该
    minibatch 的逐 token 均值，跨 minibatch/跨 rank 求和后 / nb_global = grand mean。
    """
    def matvec(blocks, vs):
        accs = [torch.zeros_like(v) for v in vs]
        for x, y in batches:
            op = SharedBatchOp(model, x, y, kind, precond)
            op.prepare(blocks)                      # ← 最贵的一步，K 个块共享
            for i, (b, v) in enumerate(zip(blocks, vs)):
                accs[i] += op.apply(b, v)
            op.release()
        if dist is not None and dist.is_initialized():
            # 拼成一条再通信：K 次小 all_reduce → 1 次
            flat = torch.cat(accs)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            off, accs = 0, []
            for v in vs:
                accs.append(flat[off:off + v.numel()])
                off += v.numel()
        return [a / nb_global for a in accs]
    return matvec
