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

| 配置                 | 方案 A (toy_model_A) | 方案 B (toy_model_B) |
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

- `model.py`         —— `ToyLlama` 模型与 `ToyLlamaConfig` 定义（方案 A/B）
- `vanilla_model.py` —— `ToyVanilla` 模型与 `ToyVanillaConfig` 定义（方案 C）
- `toy_model_A.py`   —— 方案 A 的 config 与 `build()`，直接运行可打印参数量分解
- `toy_model_B.py`   —— 方案 B 的 config 与 `build()`
- `toy_model_C.py`   —— 方案 C 的 config 与 `build()`

## 使用

```bash
cd toy_models
python3 toy_model_A.py   # 打印方案 A 参数量分解
python3 toy_model_B.py   # 打印方案 B 参数量分解
python3 toy_model_C.py   # 打印方案 C 参数量分解
```

在代码中构建模型：

```python
from toy_model_A import build
model = build()
logits, loss = model(idx, targets)   # idx/targets: (batch, seq_len) LongTensor
```
