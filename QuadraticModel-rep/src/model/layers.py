"""
Hessian / GN 分析用的参数块（block）定义。

⚠ 本模型的参数是**跨层 stack** 的（见 model.py 的 Transformer.__init__）：
    embd       (V, D)
    attn_q/k/v (L, D, H, K)
    attn_head  (L, H, K, D)
    mlp_up     (L, D, M)
    mlp_head   (L, M, D)
    head       (D, V)
named_parameters() 只有 **8** 个张量，没有 per-layer 张量。所以"第 l 层的块"不是
某个叶子张量，而是 6 个张量各自的一段连续 flat 区间（L 是首轴 → 每层那段连续）。

因此 Block 用 specs = [(param_name, start, stop)] 描述，块自身的坐标 =
各段按 specs 顺序拼接。块内坐标与"全局 flat 布局"无关，只有 name="full" 的块
才刻意按 named_parameters() 顺序铺满，用来和 spectrum_ddp.py 的全量谱对齐。

数学约定：block b 对应零填充嵌入矩阵 P_b（N × n_b），块 Hessian 为
    H_bb = P_bᵀ H P_b
它等于「只有块 b 可训练」那个子问题的 Hessian，不是任何近似。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# 带 L 首轴的、逐层可切的张量
LAYER_PARAMS = ("attn_q", "attn_k", "attn_v", "attn_head", "mlp_up", "mlp_head")
ATTN_PARAMS = ("attn_q", "attn_k", "attn_v", "attn_head")
MLP_PARAMS = ("mlp_up", "mlp_head")
# 无 L 轴的张量
GLOBAL_PARAMS = ("embd", "head")

# 常见命名 → 本仓库命名的别名（--groups 里两种写法都接受）。
# 本仓库沿用 JAX 参考树的名字，与多数 PyTorch 实现叫法不同，容易记混。
PARAM_ALIASES = {
    "attn_wq": "attn_q", "attn_wk": "attn_k", "attn_wv": "attn_v",
    "wq": "attn_q", "wk": "attn_k", "wv": "attn_v",
    "q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v",
    "attn_proj": "attn_head", "attn_wo": "attn_head",
    "o_proj": "attn_head", "attn_out": "attn_head",
    "mlp_fc": "mlp_up", "fc1": "mlp_up", "up_proj": "mlp_up",
    "mlp_proj": "mlp_head", "fc2": "mlp_head", "down_proj": "mlp_head",
}


def canon(name: str) -> str:
    """把常见别名归一到本仓库的张量名。"""
    return PARAM_ALIASES.get(name, name)


@dataclass(frozen=True)
class Spec:
    """某个叶子张量 flat 视图上的一段连续区间 [start, stop)。"""
    param: str
    start: int
    stop: int

    @property
    def numel(self) -> int:
        return self.stop - self.start


class Layer:
    def __init__(self, name: str, specs):
        self.name = name
        self.specs = tuple(specs)
        self.numel = sum(s.numel for s in self.specs)
        # 去重后的叶子张量名，保持首次出现顺序（autograd 的 inputs= 用）
        seen = []
        for s in self.specs:
            if s.param not in seen:
                seen.append(s.param)
        self.params = tuple(seen)

    def __repr__(self):
        return f"Layer({self.name}, numel={self.numel:,}, params={self.params})"

    def split(self, v):
        """块向量 (numel,) → 按 specs 顺序切成的分段列表（视图，不拷贝）。"""
        out, off = [], 0
        for s in self.specs:
            out.append(v[off:off + s.numel])
            off += s.numel
        return out

    def slices_of(self, by_name: dict):
        """从 {param_name: 与该参数同形的张量} 里取出本块各段的 flat 视图。"""
        return [by_name[s.param].reshape(-1)[s.start:s.stop] for s in self.specs]


class IndexBlock:
    """坐标子集块：由任意（可不连续的）flat 坐标集合组成的块。

    与 Layer 的区别：Layer 的 specs 是连续区间 [start,stop)，而"某几个输出神经元
    的全部输入坐标"这类子集在 flat 布局里是 stride 散点（如 mlp_up (L,D,M) 里
    单个神经元 m 是 stride=M 的 D 个点），用区间描述需要 n 段长度 1 的 Spec，
    python 循环成为瓶颈。IndexBlock 直接持有索引张量，gather/scatter 全向量化。

    indices: 有序 [(param_name, LongTensor flat 坐标)]，块内坐标 = 各段依序拼接。
    对外接口与 Layer 对齐（name/numel/params/split/slices_of），hvp_layers 的
    SharedBatchOp 可直接使用；另有 fill()（代替 _zero_filled 的区间写入）。
    ⚠ 暂不支持预条件器切段（_precond_block 走 specs），只能 precond=None 使用。
    """

    def __init__(self, name, indices):
        self.name = name
        self.indices = [(pn, torch.as_tensor(ix, dtype=torch.long).reshape(-1))
                        for pn, ix in indices]
        self.numel = sum(ix.numel() for _, ix in self.indices)
        seen = []
        for pn, _ in self.indices:
            if pn not in seen:
                seen.append(pn)
        self.params = tuple(seen)

    def __repr__(self):
        return f"IndexBlock({self.name}, numel={self.numel:,}, params={self.params})"

    def split(self, v):
        """块向量 (numel,) → 按 indices 顺序切成的分段列表（视图，不拷贝）。"""
        out, off = [], 0
        for _, ix in self.indices:
            out.append(v[off:off + ix.numel()])
            off += ix.numel()
        return out

    def slices_of(self, by_name):
        """从 {param_name: 与该参数同形的张量} 里取出本块各坐标的值。"""
        return [by_name[pn].reshape(-1)[ix.to(by_name[pn].device)]
                for pn, ix in self.indices]

    def fill(self, v_block, pmap):
        """块向量 → {param_name: 同形张量，块外为 0}（= P_b v_b）。"""
        out = {pn: torch.zeros_like(pmap[pn]) for pn in self.params}
        for seg, (pn, ix) in zip(self.split(v_block), self.indices):
            out[pn].reshape(-1)[ix.to(seg.device)] = seg
        return out


# 每输出神经元一列的张量：pname -> (d_in, d_out)。这四个张量的 flat 布局
# （层内）都是 in 维 stride=d_out、out 维 stride=1，见 model.Transformer.__init__。
_NEURON_DIMS = {
    "mlp_up":    lambda cfg: (cfg.D,          cfg.M),
    "mlp_head":  lambda cfg: (cfg.M,          cfg.D),
    "attn_v":    lambda cfg: (cfg.D,          cfg.H * cfg.K),
    "attn_head": lambda cfg: (cfg.H * cfg.K,  cfg.D),
}


def neuron_subblock(model, l, pname, neuron_ids, name=None):
    """第 l 层 pname 的「若干输出神经元 × 全部输入坐标」子块。

    块内坐标 neuron-major：idx = k*d_in + d（k=第几个选定神经元，d=输入坐标），
    这样 Hessian 热图的对角块正好是各神经元自己的 d_in×d_in 块。
    支持 mlp_up / mlp_head / attn_v / attn_head（attn_q/k 的自然 unit 是整头
    D·K 维，不是单神经元，不在此支持）。
    """
    pname = canon(pname)
    if pname not in _NEURON_DIMS:
        raise SystemExit(f"neuron_subblock 不支持 {pname}（可用 {list(_NEURON_DIMS)}）")
    d_in, d_out = _NEURON_DIMS[pname](model.cfg)
    ids = torch.as_tensor(list(neuron_ids), dtype=torch.long)
    base = l * d_in * d_out
    # (k, d) → base + d*d_out + m_k
    idx = base + torch.arange(d_in).unsqueeze(0) * d_out + ids.unsqueeze(1)  # (n_sel, d_in)
    return IndexBlock(name or f"layer{l:02d}.{pname}.n{len(ids)}",
                      [(pname, idx.reshape(-1))])


# --------------------------------------------------------------------------
# 构造器
# --------------------------------------------------------------------------
def _pmap(model):
    return dict(model.named_parameters())


def _per_layer_numel(model, pname):
    """带 L 首轴的张量：单层那一段的元素数。"""
    return _pmap(model)[pname][0].numel()


def whole_block(model, pname, name=None):
    """整个叶子张量作为一块。"""
    n = _pmap(model)[pname].numel()
    return Layer(name or pname, [Spec(pname, 0, n)])


def layer_block(model, l, params=LAYER_PARAMS, name=None):
    """第 l 层的块：params 里每个张量取第 l 层那段。"""
    specs = []
    for pn in params:
        per = _per_layer_numel(model, pn)
        specs.append(Spec(pn, l * per, (l + 1) * per))
    return Layer(name or f"layer{l:02d}", specs)


def full_block(model):
    """全模型（trivial block）。specs 按 named_parameters() 顺序铺满，
    与 spectrum_ddp.py 的 flat 布局逐坐标一致 → 可用来交叉验证两条路径。"""
    return Layer("full", [Spec(n, 0, p.numel()) for n, p in model.named_parameters()])


def attn_head_block(model, l, h, name=None):
    """第 l 层第 h 个 attention head 的块（q/k/v 的 head 切片 + attn_head 的对应切片）。

    q/k/v 形状 (L, D, H, K)：head 维 H 不是首轴 → 第 h 个 head 在 flat 里**不连续**
    （每个 D 行都有一段 K）。故用 D 段 Spec 逐段描述。
    attn_head 形状 (L, H, K, D)：H 紧跟 L → 第 h 个 head 连续，1 段即可。
    """
    cfg = model.cfg
    D, H, K = cfg.D, cfg.H, cfg.K
    specs = []
    for pn in ("attn_q", "attn_k", "attn_v"):
        per = _per_layer_numel(model, pn)      # D*H*K
        base = l * per
        for d in range(D):                    # (d, h, :) → 偏移 d*H*K + h*K
            off = base + d * H * K + h * K
            specs.append(Spec(pn, off, off + K))
    per_ah = _per_layer_numel(model, "attn_head")   # H*K*D
    base_ah = l * per_ah
    specs.append(Spec("attn_head", base_ah + h * K * D, base_ah + (h + 1) * K * D))
    return Layer(name or f"layer{l:02d}.head{h:02d}", specs)


def build_layers(model, spec: str):
    """把 --groups 字符串解析成 Block 列表。

    可用 token（逗号分隔，按给出顺序，自动去重）：
      full                 全模型
      layers               embd, head, layer00..layer{L-1}（整层）
      layer-parts          embd, head, layer{l}.attn, layer{l}.mlp
      layer-tensors        embd, head, 每层每个张量各一块（**最细**，74 块）
                           = layer{l}.{attn_q,attn_k,attn_v,attn_head,mlp_up,mlp_head}
      tensors              8 个 stacked 叶子张量各一块（跨层合并）
      embd / head / attn_q / ...        单个叶子张量
      layerNN              指定层（整层）
      layerNN.attn / .mlp  指定层的一半
      layerNN.<张量名>      指定层的单个张量，如 layer03.mlp_up
      layerNN.headHH       指定层的单个 attention head

    ⚠ 张量名对照（本仓库命名 ← 常见命名）：
        mlp_up    ← mlp_fc / fc1 / up_proj
        mlp_head  ← mlp_proj / fc2 / down_proj
        attn_q/k/v ← attn_wq / attn_wk / attn_wv
        attn_head ← attn_proj / o_proj
    """
    L = model.cfg.L
    out, seen = [], set()

    def add(b):
        if b.name not in seen:
            seen.add(b.name)
            out.append(b)

    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        tok = canon(tok)          # 顶层别名：mlp_fc → mlp_up 等
        if tok == "full":
            add(full_block(model))
        elif tok == "layers":
            add(whole_block(model, "embd"))
            add(whole_block(model, "head"))
            for l in range(L):
                add(layer_block(model, l))
        elif tok == "layer-parts":
            add(whole_block(model, "embd"))
            add(whole_block(model, "head"))
            for l in range(L):
                add(layer_block(model, l, ATTN_PARAMS, f"layer{l:02d}.attn"))
                add(layer_block(model, l, MLP_PARAMS, f"layer{l:02d}.mlp"))
        elif tok == "layer-tensors":
            # 最细粒度：每层每个张量单独一块（attn_q/k/v/head, mlp_up/head）
            add(whole_block(model, "embd"))
            add(whole_block(model, "head"))
            for l in range(L):
                for pn in LAYER_PARAMS:
                    add(layer_block(model, l, (pn,), f"layer{l:02d}.{pn}"))
        elif tok == "tensors":
            for n, _ in model.named_parameters():
                add(whole_block(model, n))
        elif tok in _pmap(model):
            add(whole_block(model, tok))
        elif tok.startswith("layer"):
            body = tok[len("layer"):]
            if "." in body:
                lstr, part = body.split(".", 1)
                l = int(lstr)
                part = canon(part)    # layer03.mlp_fc → layer03.mlp_up
                if part == "attn":
                    add(layer_block(model, l, ATTN_PARAMS, f"layer{l:02d}.attn"))
                elif part == "mlp":
                    add(layer_block(model, l, MLP_PARAMS, f"layer{l:02d}.mlp"))
                elif part in LAYER_PARAMS:
                    add(layer_block(model, l, (part,), f"layer{l:02d}.{part}"))
                elif part.startswith("head") and part != "head":
                    add(attn_head_block(model, l, int(part[len("head"):])))
                else:
                    raise SystemExit(f"未知 block token: {tok}")
            else:
                add(layer_block(model, int(body)))
        else:
            raise SystemExit(f"未知 block token: {tok}")

    if not out:
        raise SystemExit("--groups 解析后为空")
    return out


def check_partition(model, blocks):
    """诊断：blocks 是否恰好无重叠地铺满全部参数。返回 (covered, total, overlap)。"""
    total = model.n_params()
    marks = {n: bytearray(p.numel()) for n, p in model.named_parameters()}
    overlap = 0
    for b in blocks:
        for s in b.specs:
            mv = marks[s.param]
            for i in range(s.start, s.stop):
                if mv[i]:
                    overlap += 1
                mv[i] = 1
    covered = sum(sum(v) for v in marks.values())
    return covered, total, overlap
