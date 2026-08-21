# Tiny Transformer (2304 参数) 预训练 + 精确 Hessian 分析 — 计划存档

> 2026-08-21 调研完成后存档。**尚未做任何代码改动、未跑任何训练**。
> 之后继续时按本文档执行即可。

## 背景 / 目标

在 `QuadraticModel-rep` 里预训练一个极小 transformer，方便做**精确（非 Lanczos）Hessian 分析**：

| 用户口径 | repo 口径 (`TransformerConfig`) | 值 |
|---|---|---|
| vocab_size | V | 8 |
| context_length | seq_len | 32 |
| n_layer | L | 1 |
| n_head | H | 4 |
| n_embd | D | 16 |
| head_dim | K | 4 |
| mlp_hidden_dim | M | 32 |

参数总量 **2304**（embd 128 + qkv 768 + attn_head 256 + mlp 1024 + head 128）。
全量稠密 Hessian 2304×2304 fp64 ≈ 42 MB，`eigvalsh` 实测 0.8s —— **可以做精确全谱，不需要 Lanczos**。

## 环境（已确认）

- 之前 sco 挂实验用的环境：**`/data/250010020/miniconda3/envs/nanogpt`**（Python 3.12.13, torch 2.11.0+cu130）。
  - 见 `sco/submit_train_b64.sh:13`（`/data/250010020/miniconda3/envs/nanogpt/bin/torchrun`）和 `sco/run_spectrum_ddp_inner.sh:11`。
  - 容器镜像：`registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/zkx-ssh-install-g:main-20260515065803`。
  - 本机该环境可用但无 GPU；tiny 模型 CPU 跑完全够（训练几分钟量级）。

## 核心发现

### 1. vocab=8 不能用 BPE 分词器（关键结论）

**不需要、也不可能**训 `bpe_8.json`：`src/data/fineweb_data.py:50` 训练 BPE 时
`initial_alphabet=ByteLevel.alphabet()`（256 个字节符号）+ `<eot>`，byte-level BPE 词表下限 257。
真实 FineWeb 文本无法压进 8 个 token。**必须换数据源**。

### 2. monorepo 里已有现成的小词表合成数据生成器

`/data/250010020/hessian-spectrum-senmiao/data_construction/`（README 写明就是为 Hessian 异质性分析设计的）：
- `build_dataset.py`：一阶马尔可夫（bigram）token 流，`vocab_size` 任意，三个正交旋钮：
  词频 π（uniform/zipf）、predictability（Bayes 最优 loss 可精确算出）、label_mode。
- 输出双流格式：`train_x.bin / train_y.bin / val_x.bin / val_y.bin`（uint16 扁平流）+ `meta.pkl`。
- 注意：README 提到的 `configs/` 目录**实际不存在**，但 `key=value` CLI 覆盖完全够用。
- ⚠ V=8 时默认 `bandwidth_frac=0.02` 是坑（h=0.16，近确定性任务）。实测熵表（zipf_s=1.0，log8=2.079 nats）：

| bandwidth_frac | a=0.5 | a=0.8 | a=0.95 |
|---|---|---|---|
| 0.02（默认） | 1.342 | 0.707 | 0.234 |
| **0.125（推荐）** | 1.633 | **1.355** | 1.146 |
| 0.25 | 1.708 | 1.546 | 1.433 |

推荐：`zipf_s=1.0, predictability=0.8, bandwidth_frac=0.125` → Bayes floor **1.355 nats**，
unigram 基线 1.816，可学习空间 0.46 nat。

### 3. 代码现状（可复用点）

- 模型完全参数化：K、M 独立可配，V/seq_len 无硬编码；`_TINY_MODEL = TransformerConfig(D=16,L=1,M=32,H=4,K=4,V=8,seq_len=32)` 直接可用。
- 数据管线只有两个公共入口（`src/data/data_grain.py`）：`build_loaders(...)` 和 `make_hvp_batches(...)`，
  train.py 和所有谱分析脚本都走这两个签名 → 写一个同签名的合成数据 loader 即可无痛替换。
- 分析端 `load_checkpoint`（`spectrum_ddp.py:126`）从 ckpt["config"] 重建模型，V=8/seq=32 不会 break。
- **`src/analysis/verify_layers.py:27-66` 已有 `brute_hessian` / `brute_gn`**（暴力全量 Hessian/GN oracle），
  正是 tiny 模型需要的；`exact_block_hessian.py --verify` 已在用它们。
- 实测（CPU fp64, B=8, T=32）：chunked `is_grads_batched` 建全量 Hessian **7.1s/batch**（逐行 25.5s）；GN 因 V=8 也便宜。

### 4. 已知精度上限

`rmsnorm` / `apply_rope`（model.py:48-74）内部硬转 `.float()`，`.double()` 模型的 Hessian 非对称度
实测 6.9e-8（fp32 级）。可选 2 行修复（dtype 为 float64 时不降精度），或分析端对称化并记录残差。

## 实施计划（待继续时执行）

### Step 1 — 生成合成数据

```bash
PY=/data/250010020/miniconda3/envs/nanogpt/bin/python
cd /data/250010020/hessian-spectrum-senmiao/data_construction
$PY build_dataset.py vocab_size=8 freq=zipf zipf_s=1.0 \
   predictability=0.8 bandwidth_frac=0.125 label_mode=shift \
   n_train_tokens=5000000 n_val_tokens=500000 seed=1337 \
   out_dir=/data/250010020/hessian-spectrum-senmiao/QuadraticModel-rep/data/synth_v8
$PY inspect_dataset.py .../data/synth_v8   # 确认 TV~1e-10、H(y|x)≈1.355
```
（词频 π 用 zipf 还是 uniform 还是两套对照 —— **未定，收尾时用户未选**；上面命令是 zipf 版。）

### Step 2 — 新 loader `src/data/data_synth.py`

镜像 data_grain 的两个签名（keyword-only 同名）：
- `build_loaders(*, data_dir, seq_len, vocab_size, global_batch, eval_batch, device, rank, world, seed)` → `(train_iter, eval_factory, ds)`
- `make_hvp_batches(*, data_dir, seq_len, vocab_size, n_tokens, per, device, seed, rank, world)` → `(list[(x,y)], n_seqs_global)`

要点：np.memmap uint16 双流（y 从 `*_y.bin` 读，**不做平移**，参考 `toy_models/train_simpliest_model.py:102-107`）；
train 无限流（同 seed 全 rank 出同一 global batch，再按 rank 切行，对齐 `GrainBatchIterator._slice`）；
eval_factory 每次从头、顺序不重叠窗口；make_hvp_batches 用独立 `default_rng(seed)` 保证复现。
加 `__main__` 自检。

### Step 3 — 分发层（最小改动）

1. 新建 `src/data/loaders.py`：`resolve(data_source)` → 懒加载 data_grain 或 data_synth。
2. `RunConfig` 加字段 `data_source: str = "fineweb"` 和 `amp: bool = True`。
3. `train.py`：改用 `resolve(cfg.data_source).build_loaders(...)`；save_ckpt 的 config dict 加
   `data_source`、`data_dir`（checkpoint 自描述）；两处 `torch.autocast("cuda", bf16)`（150/187 行）
   改为 `amp_ctx()`（cfg.amp=False 时 nullcontext —— tiny 模型必须纯 fp32，bf16 伤 Hessian 保真度）。
4. `spectrum_ddp.py` 两处小改：`load_checkpoint` 里 `cfg.data_source = c.get("data_source","fineweb")`、
   `cfg.data_dir = c.get("data_dir", DATA_DIR)`（动态属性，返回签名不变，4 个下游 importer 全不用动）；
   `make_local_batches` 改走 `resolve(...)`。旧 ckpt 靠 `.get` 默认值兼容。
5. `compute_spectrum.py` / `run_spectrum_b64.py` 是 b64 专用，不动。

### Step 4 — preset（config.py，仿 `_SMOKE_*` 模式）

```python
_TINY_MODEL = TransformerConfig(D=16, L=1, M=32, H=4, K=4, V=8, seq_len=32)
_TINY_ADAM = RunConfig(name="tiny_adam", model=..., optim=OptimConfig(name="adam"),
    data_source="synth", data_dir=str(paths.REPO_ROOT/"data"/"synth_v8"),
    out_dir=str(paths.REPO_ROOT/"checkpoints_tiny_adam"),
    total_tokens=2_000_000, batch_size=64, seq_len=32, vocab=8,
    opt_lr=16.0, amp=False,
    ckpt_fracs=(0.02,0.05,0.10,0.20,0.35,0.50,0.75,1.00),  # 训练便宜，多存点看 Hessian 随训练演化
    ema_pct=(0.04,0.08))
```
- 977 步（2048 tok/step）。`opt_lr=16=sqrt(64)*2.0` 沿用约定，但 D=16 远离调参区间 →
  建议加 `tiny_lr{4,16,64}` 三个 sweep preset（复用 max_train_steps 机制，每个 ~1 分钟）先扫一下。
- b2_pct=0.01 在 977 步内 m 恒=20 → beta2≈0.95 常数，知悉即可。
- embd_scale=1/64 保持不变（不 fork 参数化）。
- 单进程跑：`python src/train/train.py tiny_adam`，不用 torchrun/DDP。
- （优化器只跑 Adam 还是 Adam+Muon —— **未定**；Muon 只需 replace 一行加 tiny_muon preset。）

### Step 5 — 新分析脚本 `src/analysis/exact_full_hessian.py`

CLI：`--ckpt --ema 0.04 --n_tokens 65536 --per 8 --seed 42 --chunk 64 --check-hvp 5 --precond --out ...npz`

流程：`load_checkpoint` → `model.double()` → `make_local_batches`（与其它谱脚本同口径同 seed）→
chunked `is_grads_batched` 建全量 H（7.1s/batch，256 batch ≈ 30min CPU；16384 tokens/64 batch ≈ 8min 也够，
数据分布只有 64 个自由参数，收敛很快）→ GN 用 `gauss_newton_vector_product` 逐列 →
对称化并记录非对称残差 → `eigvalsh` → 存 npz（eigs_H, eigs_GN, H, G, param_group_slices, 元数据）。

内置校验：单 batch 与 `verify_layers.brute_hessian`/`brute_gn` allclose；随机向量 `H@v` 对拍
`hvp.hessian_vector_product`。`--precond` 时用 load_checkpoint 返回的对角预条件器额外算 PHP 谱。
9 个 ckpt（p0…p100）全扫，逐 ckpt 并行。

### Step 6 — 端到端验证顺序

1. build_dataset + inspect（TV、H(y|x)）
2. `python -m src.data.data_synth <dir>` loader 自检
3. `python src/config.py`（现有 preset 无回归）
4. lr sweep → `train.py tiny_adam`（val_loss 逼近 1.355，entropy 从 log8=2.079 降）
5. `exact_block_hessian.py --verify`（原有 oracle 不回归）
6. `exact_full_hessian.py --check-hvp 5`
7. tiny ckpt 过一遍现有管线：`spectrum_ddp.py --cpu`（Lanczos 端点 vs 精确谱端点对拍）、
   `exact_block_hessian.py`（块 = H[slice,slice] 对拍）
8. fineweb 路径无回归（smoke preset 仍可 import/跑）

## 风险备忘

- RoPE K=4 → half=2，仅 2 个频率，第二维 32 位置内只转 0.31 rad —— 位置信号弱，对 bigram 任务无所谓；可选 rope_freq 调到 ~100，但为可比性建议保持默认。
- head 零初始化 → p0 时 logits≡0、部分 Hessian 块精确为 0，是特性不是 bug（exact_block_hessian.py:305 注释有记载）。
- `ignore_index=-1` 不影响（token 0-7 无 padding）。
- uint16 存 V=8 浪费一半，但保持双流格式兼容，共 20MB，无所谓。

## 未决问题（继续时先问）

1. 词频 π：zipf s=1.0 / uniform / 两套对照？
2. 优化器：只 Adam，还是 Adam+Muon 各训一个？
3. rmsnorm/RoPE 的 fp64 上限：改 2 行模型代码，还是分析端对称化了事？
