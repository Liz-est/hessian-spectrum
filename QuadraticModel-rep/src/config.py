"""
QuadraticModel-rep 轻量单文件 config：统一管理模型 config、batch_size、训练超参、优化器选择。

用法：
  from src.config import load, build_optimizer
  cfg = load("b64_muon")               # 或 "b64_adam"
  model = Transformer(cfg.model)
  opt = build_optimizer(model, cfg, total_steps)

在 config 里切换优化器只需改 preset 的 optim.name（"adam"|"muon"）。checkpoint 会记录
optim_name，谱分析 spectrum_ddp.py 据此自动切换 preconditioned-Hessian 口径。

设计取舍（比 toy_models/config 精简）：模型 config 直接复用 model.TransformerConfig，
不重定义；PRESETS 是普通 dict，preset 名即 run 名。
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field, replace
from typing import Tuple

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import paths
from src.model.model import TransformerConfig
from src.optim.opt import CompletePAdam, CompletePMuon


# ---------------------------------------------------------------------------
# 优化器超参（union：各优化器只读自己相关字段）
# ---------------------------------------------------------------------------
@dataclass
class OptimConfig:
    name: str = "adam"                       # "adam" | "muon"
    # ---- Adam（CompletePAdam）----
    b1: float = 0.9
    b2_pct: float = 0.01
    eps: float = 1e-8
    # ---- Muon（CompletePMuon, Moonlight NS5）----
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_rms_match: bool = False             # True 则叠 Moonlight 0.2·√max(m,n) 因子


# ---------------------------------------------------------------------------
# 整个 run 的 config（模型 + 数据 + 训练 + 优化器）
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    name: str = "b64_adam"
    model: TransformerConfig = field(default_factory=lambda: TransformerConfig(V=8192, seq_len=1024))
    optim: OptimConfig = field(default_factory=OptimConfig)
    # 数据
    data_dir: str = str(paths.DATA_DIR)
    out_dir: str = str(paths.CKPT_DIR)
    total_tokens: int = 3_000_000_000
    batch_size: int = 64                     # 全局 batch（跨所有 GPU），可调
    seq_len: int = 1024
    vocab: int = 8192
    # 优化调度（cosine，与原 train.py 一致）
    opt_lr: float = 16.0                      # eta = sqrt(64)*2.0
    warmup_pct: float = 0.1
    init_v: float = 0.1
    peak_v: float = 1.0
    end_v: float = 0.1
    ema_pct: Tuple[float, ...] = (0.04, 0.08)
    ckpt_fracs: Tuple[float, ...] = (0.10, 0.50, 1.00)
    seed: int = 0
    eval_every_frac: float = 1 / 30
    eval_batches: int = 40
    # ---- lr 扫描/短跑用（不影响正式训练，默认关）----
    # schedule 仍按 total_tokens 的 full steps（公平对比），仅循环提前停 + 密集 eval + 不存 ckpt。
    max_train_steps: int = 0        # >0 则只跑这么多步（schedule 不变）
    eval_every_steps: int = 0       # >0 则每这么多步 eval 一次（覆盖 eval_every_frac）

    def schedule_kwargs(self):
        return dict(warmup_pct=self.warmup_pct, init_value=self.init_v,
                    peak_value=self.peak_v, end_value=self.end_v)


# ---------------------------------------------------------------------------
# 优化器构造：按 optim.name 分派，都吃 model.lr_groups()
# ---------------------------------------------------------------------------
def build_optimizer(model, cfg: RunConfig, total_steps: int):
    """返回 CompletePAdam 或 CompletePMuon。两者都吃 (name,param,pre,post) 四元组。"""
    o = cfg.optim
    lr_groups = model.lr_groups()
    name = o.name.lower()
    if name == "adam":
        return CompletePAdam(lr_groups, lr=cfg.opt_lr, total_steps=total_steps,
                             b1=o.b1, b2_pct=o.b2_pct, eps=o.eps,
                             schedule_kwargs=cfg.schedule_kwargs())
    if name == "muon":
        return CompletePMuon(lr_groups, lr=cfg.opt_lr, total_steps=total_steps,
                             momentum=o.muon_momentum, nesterov=o.muon_nesterov,
                             ns_steps=o.muon_ns_steps, rms_match=o.muon_rms_match,
                             schedule_kwargs=cfg.schedule_kwargs())
    raise ValueError(f"未知优化器 name={o.name!r}（可选 adam|muon）")


# ---------------------------------------------------------------------------
# PRESETS：preset 名即 run 名
# ---------------------------------------------------------------------------
_ADAM = RunConfig(
    name="b64_adam",
    optim=OptimConfig(name="adam"),
    out_dir=str(paths.CKPT_DIR),
)

_MUON = replace(
    copy.deepcopy(_ADAM),
    name="b64_muon",
    optim=OptimConfig(name="muon"),
    out_dir=str(paths.CKPT_DIR_MUON),
)

PRESETS = {
    "b64_adam": _ADAM,
    "b64_muon": _MUON,
}

# ---- smoke presets：tiny 模型 + tiny tokens，仅用于端到端验证（GPU roundtrip）----
# 保持真实 vocab=8192 使 data_grain 管线直接可用；D/L/seq 都调小求快。
_SMOKE_MODEL = TransformerConfig(D=64, L=2, M=128, H=4, K=16, V=8192, seq_len=64)
_SMOKE_ADAM = RunConfig(
    name="smoke_adam",
    model=copy.deepcopy(_SMOKE_MODEL),
    optim=OptimConfig(name="adam"),
    out_dir=str(paths.TEST_OUT_DIR / "smoke_adam"),
    total_tokens=2_000_000, batch_size=8, seq_len=64, opt_lr=1.0,
    ema_pct=(0.04,), ckpt_fracs=(1.00,),
)
_SMOKE_MUON = replace(copy.deepcopy(_SMOKE_ADAM), name="smoke_muon",
                      optim=OptimConfig(name="muon"),
                      out_dir=str(paths.TEST_OUT_DIR / "smoke_muon"))
PRESETS["smoke_adam"] = _SMOKE_ADAM
PRESETS["smoke_muon"] = _SMOKE_MUON

# ---- lr 扫描 presets：真实 b64 模型 + Muon，仅 opt_lr 不同 ----
# schedule 仍按 full 45776 步（total_tokens=3e9），只跑 max_train_steps 步做早段对比。
# Adam(opt_lr=16) 基准：step1526→3.58, step3052→3.36（都在 warmup 内）。
# 非 Moonlight 版 Muon 经验 lr ≈ 10×Adam，以 160 为中心两边扫。
_SWEEP_STEPS = 3200
def _muon_lr_preset(lr):
    tag = f"sweep_muon_lr{lr:g}"
    return replace(
        copy.deepcopy(_MUON),
        name=tag,
        opt_lr=float(lr),
        out_dir=str(paths.TEST_OUT_DIR / tag),
        ema_pct=(0.04,), ckpt_fracs=(),        # 扫描不存 ckpt
        max_train_steps=_SWEEP_STEPS,
        eval_every_steps=200,
    )

for _lr in (80, 160, 240, 320):
    _p = _muon_lr_preset(_lr)
    PRESETS[_p.name] = _p


def load(name: str) -> RunConfig:
    if name not in PRESETS:
        raise KeyError(f"未知 preset {name!r}；可选 {list(PRESETS)}")
    return copy.deepcopy(PRESETS[name])


if __name__ == "__main__":
    from src.model.model import Transformer
    for nm in ("b64_adam", "b64_muon"):
        cfg = load(nm)
        model = Transformer(cfg.model)
        steps = cfg.total_tokens // (cfg.batch_size * cfg.seq_len)
        opt = build_optimizer(model, cfg, steps)
        print(f"{nm}: optim={cfg.optim.name} batch={cfg.batch_size} "
              f"steps={steps} out={cfg.out_dir.split('/')[-1]} "
              f"-> {type(opt).__name__}")
        del model, opt
