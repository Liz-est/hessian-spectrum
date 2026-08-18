"""
严格复现 QuadraticModel 的数据采样。采样内核用本地副本 fineweb_data.FineWeb
（= QuadraticModel/data.py 整份复制，不 import 原版），本文件只做与框架无关的封装：

  把 grain 产出的 numpy batch 迭代器包成 PyTorch 的 (x, y) 张量流，并在 DDP 下
  按 rank 切 batch 轴的行。

对应关系（见 QuadraticModel/pretrain.py）：
  build_loaders 里的 train/eval 两条 ds.build(...)  ==  pretrain.py 行102/103
  GrainBatchIterator._slice（按 rank 切 batch 轴）  ==  pretrain.py 的 P("data") 行分片
  train_iter / eval_it 的迭代                       ==  pretrain.py 的 iter(train_ds)/iter(eval_ds)

采样内核（fineweb_data.FineWeb.build）一字不改，语义为：
  对 train=shards[:-1]（eval=shards[-1:]）做 seed 全局文档 shuffle → 逐文档分词并
  追加 <eot> → ConcatThenSplit 切成连续 (seq_len+1) 窗口 → 顺序 batch。

为什么复用同一份 grain 代码而非重写：grain 的 batch 内容只由 grain 库 + seed 决定，
与下游是 JAX 还是 PyTorch 无关（grain 既不 import jax 也不 import torch）。因此只要
构造管线（shuffle/repeat/concat-split/batch 顺序）与原版逐行一致、seed 相同，产出的
batch 就与 JAX run 逐个一致；框架相关的只有「batch 的行如何分到各卡」这一层，即下面的
按 rank 行切片。

DDP 对齐：原版单进程 grain 出一条 batch=64 的全局流，JAX 用 mesh + P("data") 把每个
64 行 batch 按第 0 轴切给 8 张卡（卡 d 拿第 [d*8:(d+1)*8] 行）。rep 用 PyTorch DDP：
8 个进程各自迭代**同一条 grain 流**（同 seed → 每个 rank 拿到的 64 行 batch 逐个相同），
rank r 只取自己那段行 [r*local_bs:(r+1)*local_bs] 喂模型，backward 时 DDP all-reduce
梯度求平均，数学上等于原版那个 64 行 batch 的梯度。行布局与 P("data") 逐行对齐。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

# 采样内核：本地副本（不 import 原版 QuadraticModel）。
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.fineweb_data import FineWebParquet  # noqa: E402


def make_fineweb(seq_len: int, vocab_size: int, data_dir: str | None = None):
    """构造 parquet 源的 FineWebParquet。

    ⚠ 用 parquet 源（非原版 arrayrecord）：grain 对 parquet 只有顺序 IterDataset，
    shuffle 用滑窗近似而非原版全局索引 shuffle，故文档顺序与原版不同、非 bit 级复现
    （配额受限下的取舍，见 fineweb_data.FineWebParquet 说明）。tokenizer 复用同一份
    bpe_8192.json（sha1 与原版一致）。"""
    ds = FineWebParquet(
        seq_len=seq_len,
        vocab_size=vocab_size,
        parquet_dir=data_dir,
    )
    # 触发 tokenizer 加载并校验 <eot> id。
    assert ds.tokenizer.token_to_id("<eot>") == 0, "bpe_8192 的 <eot> 必须是 id 0"
    return ds


class GrainBatchIterator:
    """把 grain 的 build() 输出迭代成 PyTorch (x, y) 张量。

    grain 每个元素是 ((x, y), n_bytes)，x/y 形状 (global_batch, seq_len)，int32。
    DDP 下取本 rank 的行切片 [rank*local_bs:(rank+1)*local_bs]。
    """

    def __init__(self, grain_ds, device, rank=0, local_bs=None):
        self._it = iter(grain_ds)
        self.device = device
        self.rank = rank
        self.local_bs = local_bs

    def __iter__(self):
        return self

    def _slice(self, arr):
        if self.local_bs is None:
            return arr
        s = self.rank * self.local_bs
        return arr[s:s + self.local_bs]

    def __next__(self):
        (x, y), _n_bytes = next(self._it)
        x = np.ascontiguousarray(self._slice(np.asarray(x)))
        y = np.ascontiguousarray(self._slice(np.asarray(y)))
        xt = torch.from_numpy(x.astype(np.int64)).to(self.device, non_blocking=True)
        yt = torch.from_numpy(y.astype(np.int64)).to(self.device, non_blocking=True)
        return xt, yt


def build_loaders(
    *,
    data_dir: str,
    seq_len: int,
    vocab_size: int,
    global_batch: int,
    eval_batch: int,
    device,
    rank: int = 0,
    world: int = 1,
    seed: int = 0,
):
    """返回 (train_iter, eval_factory, ds)。

    - train_iter: 无限流（parquet 源，滑窗 shuffle + repeat）。
    - eval_factory(): 每次调用返回一个新的、有限的 eval 迭代器（shuffle=False,
      repeat=False，与原版 eval_ds 每次 `iter(eval_ds)` 从头一致）。
    """
    assert global_batch % world == 0, f"global_batch {global_batch} 不整除 world {world}"
    local_bs = global_batch // world

    ds = make_fineweb(seq_len, vocab_size, data_dir)

    train_grain = ds.build("train", global_batch, seed, shuffle=True, repeat=True)
    train_iter = GrainBatchIterator(train_grain, device, rank=rank, local_bs=local_bs)

    def eval_factory():
        # 原版 eval：seed=0, shuffle=False, repeat=False，eval_batch 为全局大小。
        eval_grain = ds.build("eval", eval_batch, seed=0, shuffle=False, repeat=False)
        return GrainBatchIterator(
            eval_grain, device, rank=rank, local_bs=eval_batch // world
        )

    return train_iter, eval_factory, ds


def make_hvp_batches(
    *,
    data_dir: str,
    seq_len: int,
    vocab_size: int,
    n_tokens: int,
    per: int,
    device,
    seed: int = 0,
    rank: int = 0,
    world: int = 1,
):
    """为 Hessian/GN 谱的 HVP 采样一组固定 minibatch，用与训练相同的 grain 采样口径。

    对齐原版 analysis/spectrum/run_dist.py：从**训练同一条 grain 流**
    （build("train", ..., seed, shuffle=True, repeat=True)）顺序取样本，
    共取 data_batch_size 条序列做 HVP 平均：

        data_batch_size = per * (n_tokens // seq_len // per)

    与原版 `batch_size * (n_tokens // seq_len // batch_size)` 同式（这里 batch_size
    即每个 HVP minibatch 的序列数 per）。ConcatThenSplit 先把文档流拼接、再切成连续
    (seq_len+1) 窗口，窗口序列与 batch 大小无关，故用 batch_size=per 直接产出即为一个
    HVP minibatch。

    ⚠ parquet 源用滑窗 shuffle（非原版全局索引 shuffle），文档顺序与原版不同、非 bit
    级复现；此外以下两点也无法在 rep 侧复刻、故意从流首开始：
      1. 原版 train_iter.set_state(...) 把数据指针恢复到该 checkpoint 训练时的位置；
         rep 训练未落 grain 迭代器状态，只能从 seed 流首取（同一批确定性样本）。
      2. 原版还用 hessian_frob2 的 filtered_data.csv 只保留高曲率样本子集；
         rep 无该预处理产物，取连续样本（不做 frob2 过滤）。

    DDP：world>1 时每个 rank 迭代同一条流，再按 rank 切 minibatch 的行
    [rank*local_per:(rank+1)*local_per]，与 build_loaders 的行分片一致。

    返回 (batches, n_seqs_global)：
      batches — list[(x, y)]，每个 (per_local, seq_len) int64 张量在 device 上；
      n_seqs_global — 全局总序列数（= per * n_minibatch），供 HVP 求和后归一化。
    """
    assert per % world == 0, f"per {per} 不整除 world {world}"
    local_per = per // world

    n_minibatch = max(1, n_tokens // seq_len // per)
    n_seqs_global = per * n_minibatch

    ds = make_fineweb(seq_len, vocab_size, data_dir)
    grain_ds = ds.build("train", per, seed, shuffle=True, repeat=True)
    it = GrainBatchIterator(grain_ds, device, rank=rank, local_bs=local_per)

    batches = []
    for _ in range(n_minibatch):
        batches.append(next(it))
    return batches, n_seqs_global
