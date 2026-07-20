# Toy Models

三个单层 toy model：

- **方案 A / B**：LLaMA-style decoder（RMSNorm + RoPE + SwiGLU，无 bias）
- **方案 C**：vanilla decoder-only Transformer（Post-LayerNorm + 固定 sinusoidal PE + ReLU FFN，带 bias）

三者词表均为 `V = 1024`，Embedding 与 LM Head 均**不共享权重**（untied）。

---

## 方案 A / B：LLaMA-style decoder

### 通用配置

- 单层（`n_layer = 1`）LLaMA-style decoder
- 词表 `V = 1024`
- Embedding 与 LM Head **不共享权重**（untied）
- RMSNorm + RoPE + SwiGLU FFN，全部无 bias

### 两组配置

| 配置                 | 方案 A (llama_A) | 方案 B (llama_B) |
| -------------------- | -------------------: | -------------------: |
| Hidden size `d`      |                  160 |                  256 |
| Attention heads `h`  |                    5 |                    4 |
| Head dimension `d_head` |                 32 |                   64 |
| FFN size `d_ff`      |                  448 |                  680 |
| **总参数量**         |         **0.646M**   |         **1.309M**   |
| Embedding + LM Head  |         0.328M       |         0.524M       |
| └ 占总参数比例       |         50.8%        |         40.0%        |
| Transformer 主体     |         0.318M       |         0.785M       |
| └ 占总参数比例       |         49.2%        |         60.0%        |


### 参数量公式

```
N_embed+head  = 2 · V · d
N_transformer = 4 · d² + 3 · d · d_ff + 3 · d
```

其中 `4d²` 为 attention 的 wq/wk/wv/wo（此处 `h · d_head = d`），
`3 · d · d_ff` 为 SwiGLU 的 gate/up/down，`3d` 为 2 个 block 内 RMSNorm + 1 个 final RMSNorm。

验证：
- 方案 A：`4·160² + 3·160·448 + 3·160 = 317,920`；`2·1024·160 = 327,680`
- 方案 B：`4·256² + 3·256·680 + 3·256 = 785,152`；`2·1024·256 = 524,288`

---

## 方案 C：vanilla decoder-only Transformer

### 配置

| 配置                        | 推荐设置              |
| --------------------------- | --------------------- |
| Transformer layers `L`      | 1                     |
| Vocabulary size `V`         | 1024                  |
| Hidden size `d`             | 192                   |
| Attention heads `h`         | 6                     |
| Head dimension `d_head`     | 32                    |
| FFN size `d_ff`             | 1024                  |
| Context length              | 128                   |
| Attention type              | Causal self-attention |
| Position encoding           | Fixed sinusoidal      |
| Normalization               | Post-LayerNorm        |
| FFN activation              | ReLU                  |
| Weight tying                | False                 |
| Linear bias                 | True                  |
| Dropout                     | 0                     |
| Attention dropout           | 0                     |

### 参数量

| 参数部分            |   参数量 |    占比 |
| ------------------- | -------: | ------: |
| Token Embedding     |   0.197M |   20.9% |
| LM Head             |   0.198M |   21.1% |
| Embedding + LM Head |   0.394M |   42.0% |
| Self-attention      |   0.148M |   15.7% |
| FFN                 |   0.394M |   42.0% |
| LayerNorm           |   0.001M |    0.1% |
| **Transformer 主体**|   0.543M |   57.9% |
| **总参数量**        |   0.939M |    100% |

LM Head 的 bias 计入参数量；固定 sinusoidal position encoding 无可训练参数。

### 参数量公式

```
N_embed+head  = 2·V·d + V           = 2·1024·192 + 1024 = 394,240
N_transformer = 4d² + 2·d·d_ff + 9d + d_ff = 543,424
N_total       = 394,240 + 543,424   = 937,664 ≈ 0.938M
```

其中 `4d²` 为 attention 的 qkvo 权重，`9d` = `4d`（qkvo bias）+ `d`（FFN 输出 bias）+ `4d`（2 个 post-LayerNorm 的 weight+bias）；
`2·d·d_ff` 为 FFN 的两层权重，`d_ff` 为 FFN 第一层 bias。

---

## 文件说明

- `llama_model.py`   —— `ToyLlama` 模型与 `ToyLlamaConfig` 定义（方案 A/B）
- `vanilla_model.py` —— `ToyVanilla` 模型与 `ToyVanillaConfig` 定义（方案 C）
- `llama_A.py`       —— 方案 A 的 config 与 `build()`，直接运行可打印参数量分解
- `llama_B.py`       —— 方案 B 的 config 与 `build()`
- `vanilla_transformer.py`   —— 方案 C 的 config 与 `build()`
- `train_vanilla_transformer.py`   —— 训练方案 C（loss 曲线 + 4 个 checkpoint），支持 8 卡 DDP
- `hessian_toy.py`   —— 精确 per-unit Hessian 块（按头/按神经元/按 token）+ 特征分解 + hetero 距离
- `analyze_vanilla.py`     —— 读取 checkpoint，出 Hessian 谱 / hetero 热图 / 演变图，支持 8 卡分片
- `submit_sco_vanilla.py` —— 提交 8×H100 SCO 作业（DDP 训练 → 8 卡分片分析）
- `run_C.sh`         —— 一键（单机）：训练 + 分析

## 使用

```bash
cd toy_models
python3 llama_A.py   # 打印方案 A 参数量分解
python3 llama_B.py   # 打印方案 B 参数量分解
python3 vanilla_transformer.py   # 打印方案 C 参数量分解
```

在代码中构建模型：

```python
from llama_A import build
model = build()
logits, loss = model(idx, targets)   # idx/targets: (batch, seq_len) LongTensor
```

---

## 方案 C：训练 + Hessian 异质性分析

用合成 bigram 数据训练方案 C，并分析其 Hessian 谱与「异质性（heterogeneity）」
随训练的演变。

### 数据

`<repo-root>/data/synth_uniform_balanced_V1024/`（`data_construction` 生成的双流
`*_x.bin`/`*_y.bin`，词表 **V=1024**，与方案 C 对齐；token id ∈ [0,1023]）。
这是一阶马尔可夫链（predictability=0.8）合成数据，最优 loss 由链的条件熵率决定。

### 一键运行

```bash
cd toy_models
bash run_C.sh                       # 训练 8000 iters + 全部分析（CPU 约 1 小时）
# 或分两步：
python3 train_vanilla_transformer.py              # 训练，存 checkpoint + loss 曲线
python3 analyze_vanilla.py                # 分析 4 个 checkpoint，出全部图
```

> 注：本机 96 核，脚本已把线程数限制为 `OMP_NUM_THREADS=8`——否则线程争用会让这个
> 小模型慢约 30 倍。

### 训练配置（`train_vanilla_transformer.py`）

| 项 | 值 |
| --- | --- |
| optimizer | AdamW (β=0.9/0.95, wd=0.1) |
| lr | 6e-4，cosine decay，warmup 200 |
| batch / context | 64 × 128 |
| max_iters（=100%） | 8000 |
| checkpoint 记录点 | **init (0%) / p10 (10%) / p50 (50%) / p100 (100%)** |

CLI 可覆盖任意参数，如 `python3 train_vanilla_transformer.py --max_iters=4000`。

### Hessian 方法：精确 per-unit 块（不用 SLQ）

模型很小，因此**不做随机 Lanczos（SLQ）**，而是对每个「单元（unit）」**精确构造
Hessian / Gauss–Newton 块并用 `torch.linalg.eigvalsh` 特征分解**（有 GPU 时在 GPU 上）。
「单元」指某个权重矩阵按什么粒度切块——**不同层用不同粒��**：

| 层 | 分块粒度 | 单元数 | 块尺寸 |
| --- | --- | --- | --- |
| `embedding`（`tok_emb`） | **按 token** | V（默认前 256） | d×d = 192×192 |
| Q（`attn.wq`） | **按注意头** | h = 6 | (d_head·d)×(d_head·d) = 6144² |
| K（`attn.wk`） | **按注意头** | h = 6 | 6144² |
| V（`attn.wv`） | **按输出神经元** | 192 | 192² |
| `attn.proj`（`attn.wo`） | **按输出神经元** | 192 | 192² |
| `mlp.fc`（`c_fc`） | **按输出神经元** | 1024 | 192² |
| `mlp.proj`（`c_proj`） | **按输出神经元** | 192 | 1024² |
| `lm_head` | **按 token（=类别）** | V（默认前 256） | 192² |

三种块的精确构造（均利用「输出对该单元的权重是线性的」，二阶项严格为零）：

- **按输出神经元**（线性层 `y = W x`，第 i 行 `w_i`，`y_i = w_iᵀx`）：
  `H_i = (1/N) Σ_t s_{i,t} x_t x_tᵀ`，`s_{i,t} = (∂L/∂y_{i,t})²`（empirical-Fisher / GN 曲率，PSD）。
- **按注意头**（头 h 拥有 W 的第 `[h·d_head:(h+1)·d_head]` 行，展平权重 `vec(W_h)`，
  每 token 梯度向量 = `x_t ⊗ g_{h,t}`）：
  `H_h = (1/N) Σ_t (x_t x_tᵀ) ⊗ (g_{h,t} g_{h,t}ᵀ)`，以 `UᵀU/N`（行 `u_t = x_t ⊗ g_{h,t}`）构造。
- **按 token — `lm_head`（CE，精确）**：类别 k 即输出神经元 k，
  `H_k = (1/N) Σ_t p_{k,t}(1−p_{k,t}) x_t x_tᵀ`（与 vision 的 `ce_last_layer_hessian_blocks` 一致）。
- **按 token — `embedding`（Fisher）**：token id v 拥有嵌入行 `e_v`，
  `H_v = (1/N_v) Σ_{t: x_t=v} g_t g_tᵀ`，`g_t = ∂L/∂(该位置的嵌入输出)`，只在 v 实际出现的位置上累加。

每个块特征分解后：(1) 汇总成该层的特征值谱 → **ESD 图**；(2) 把每个单元的谱在
log-特征值域直方图化成概率分布，两两计算 **Symmetric KL** 与 **JS distance** →
**hetero 矩阵/热图**（`pairwise_matrix` 在 GPU 上向量化）。

分析的层（`analyze_vanilla.py` 中 `default_layer_spec`）：
`embedding, attn_wq, attn_wk, attn_wv, attn_proj, mlp_fc, mlp_proj, lm_head`（共 8 层）。

### 产物

```
runs/toy_C/
  loss_curve.png            loss 曲线
  loss_log.csv              iter, train_loss, lr
  ckpt_{init,p10,p50,p100}.pt

files/toy_C/
  <tag>/                    tag ∈ {init, p10, p50, p100}
    spectrum_<layer>.png / _log.png     8 层各自的 Hessian 谱（ESD），layer ∈ 上表 8 项
    hetero_<layer>_{skl,js}.png         8 层各自的 hessian hetero 热图（SKL + JS）
    eigs_<layer>.npy / hetero_<layer>_{skl,js}.npy / summary_<layer>.json   原始数据
  evolution_skl.png / evolution_js.png  hetero 均值随 epoch 演变
                                        (init/10%/50%/100%，每层一条线)
  all_summary.json          所有 checkpoint × 所有层的 hetero 均值汇总
```

- **loss 曲线**：`runs/toy_C/loss_curve.png`
- **Hessian-spectrum 图（last layer & hidden layer 都有）**：各 `<tag>/spectrum_<layer>.png`
- **last layer hessian hetero 图**：各 `<tag>/hetero_lm_head_{skl,js}.png`（按 token 分块）
- **hidden layer neuron hessian hetero 图**：各 `<tag>/hetero_{attn_wq,attn_wk,attn_wv,attn_proj,mlp_fc,mlp_proj}_{skl,js}.png`
- **随 epoch 演变过程图（记录 init/10%/50%/100%）**：`files/toy_C/evolution_{skl,js}.png`
  （metric 为 symmetric KL 与 JS distance 各一张）

### 8 卡并行（SCO）

`submit_sco_vanilla.py`（参考 `language_models/submit_sco.py`）提交单个 8×H100 作业，
两阶段串行执行：

1. **训练**：`torchrun --nproc_per_node=8 train_vanilla_transformer.py`（DDP，每卡 batch=64，
   有效 batch=512；rank 0 评估/记录/写 checkpoint）。
2. **分析**：`torchrun --nproc_per_node=8 analyze_vanilla.py`——把 `(checkpoint × 层)` 共
   4×8=32 个工作项按 rank strided 切分到 8 张卡上并行做精确特征分解；结束后 rank 0
   统一渲染所有图。

```
python3 submit_sco_vanilla.py          # 需确认
python3 submit_sco_vanilla.py --yes    # 直接提交
```

单机也可直接跑（CPU 单进程或单卡）：`python3 train_vanilla_transformer.py && python3 analyze_vanilla.py`。

