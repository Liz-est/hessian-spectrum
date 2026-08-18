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
