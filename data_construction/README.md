# 合成语言数据构造（for Hessian 异质性分析）

这个模块用来构造**可控的合成语言数据**，方便研究词频不均衡 / 任务难度 /
输出标签分布等因素如何影响 Hessian 的异质性（heterogeneity），以及 SGD vs Adam
在这些数据上的差异。

数据是一个**一阶马尔可夫链（bigram）**生成的 token 序列。核心设计是把三个因素
做成**互相正交、可独立调节的旋钮**，从而做干净的受控实验。

## 三个正交旋钮

| 旋钮 | 参数 | 控制什么 | 对 Hessian 的影响 |
|------|------|----------|-------------------|
| **1. 词频分布 π** | `freq`, `zipf_s`, `real_counts_path` | 哪些 token 频繁 / 稀有（平稳分布） | 直接决定 softmax / unembedding 头的**类别不均衡**。balance vs imbalance 数据库靠这个 |
| **2. 可预测性（难度）** | `predictability` ∈ [0,1] | 给定当前 token，下一个 token 有多确定 | 决定 loss 能压多低、landscape 有多病态。与 π **完全解耦** |
| **3. 输出标签模式** | `label_mode`, `shift` | y 如何由 x 得到 | 默认 `y = x 平移一位`；预留了让 output 独立于 input 的接口 |

### 为什么词频和难度能解耦（关键构造）

转移矩阵 P 是两个**都以 π 为精确平稳分布**的核的凸组合：

```
P = (1 - a) · Π_indep  +  a · B          其中 a = predictability
```

- `Π_indep[i,j] = π[j]`：每行都等于 π。平稳分布严格是 π，且熵最大 → 下一个 token
  除边际外不可预测（**最难**，模型只能学到 unigram 词频）。
- `B`：Metropolis–Hastings 核，用一个集中的提议分布构造。由细致平衡，**对任意提议
  它的平稳分布都严格是 π**，但每行很尖锐（低熵）→ 下一个 token 高度可预测（**最易**）。

因为两个核共享平稳分布 π，它们的凸组合平稳分布也严格是 π。所以**无论 `predictability`
取何值，词频 π 都精确不变**。代码里已用幂迭代验证：改 `predictability` 时
`TV(实际平稳分布, π) ≈ 1e-10`。

> 术语对照：`predictability` 高 = "温度"低（P 行尖锐）；`predictability` 低 = "温度"高（P 行平坦）。

## 文件

| 文件 | 作用 |
|------|------|
| `transition.py` | 核心数学：`make_pi`（词频）、`build_transition`（构造 P）、马尔可夫采样、诊断函数 |
| `build_dataset.py` | 编排：采样 token 流、切 train/val、写双流 `*_x.bin`/`*_y.bin` + `meta.pkl` |
| `configs/` | 示例配置：`uniform_balanced.py`、`zipf_imbalanced.py`、`real_aligned.py` |
| `inspect_dataset.py` | 验证：经验词频 vs 目标 π、每行熵分布、P 局部热图，输出 `inspect.png` |

## 用法

```bash
cd data_construction

# 生成 balance 数据（uniform 词频）
python build_dataset.py configs/uniform_balanced.py

# 生成 imbalance 数据（zipf 词频，难度与 balanced 相同 -> 干净对比）
python build_dataset.py configs/zipf_imbalanced.py

# 命令行覆盖任意参数
python build_dataset.py configs/zipf_imbalanced.py zipf_s=1.5 predictability=0.3 out_dir=data/synth_hard

# 检查生成结果
python inspect_dataset.py data/synth_zipf_imbalanced
```

### 做受控实验（controlled sweep）

```bash
# 固定 imbalance 词频，只扫难度 -> 看 Hessian 随可预测性怎么变
for p in 0.0 0.3 0.6 0.9; do
  python build_dataset.py configs/zipf_imbalanced.py predictability=$p out_dir=data/synth_p$p
done

# 固定难度，只扫词频不均衡程度 -> 看 Hessian 随 imbalance 怎么变
for s in 0.0 0.5 1.0 1.5; do
  python build_dataset.py configs/zipf_imbalanced.py zipf_s=$s out_dir=data/synth_s$s
done
```

## 磁盘格式（双流）

每个数据集目录包含：

```
train_x.bin   uint16   输入 token
train_y.bin   uint16   目标 token（默认 = train_x 平移一位）
val_x.bin     uint16
val_y.bin     uint16
meta.pkl      dict: vocab_size, pi, P, config, seed, label_mode, ...
```

**为什么用双流**：原始 NanoGPT 格式只存单条 token 流，在 `get_batch` 里用
"输入平移一位"临时算出目标——这把 "output == 平移后的 input" 写死了。把 x 和 y 存成
两条独立的流后，默认行为完全一样（y 就是 x 平移），但为将来"指定一个**不是 input 平移**
的输出分布"留好了接口。

## 接入训练代码

`train_gpt2.py` 和 `hessian_spectrum.py` 已改造成**自动识别格式**、且**向后兼容**：

- 若数据目录里存在 `train_x.bin` → 走双流，x/y 各自读，y 不做平移。
- 否则 → 回退到旧的单流 `train.bin` + "平移一位"逻辑（openwebtext 等不受影响）。

在 config 里把 `dataset` 指到你生成的目录名即可，例如
`config/train_gpt2_small.py` 中设 `dataset = 'synth_zipf_imbalanced'`
（数据目录相对 `language_models/data/`）。

## 未来：让 output 独立于 input

目前 `label_mode='shift'`（默认，y = x 平移）。要构造 output ≠ input 平移的情况，
**只需改一个地方**：`build_dataset.py` 里的 `build_targets(x, cfg)` 函数。里面已写好
注释和示例分支，例如：

- `relabel`：一个固定的 token→token 重映射（映射本身可从某个指定分布抽取）；
- `sample`：从一个**独立的输出核** `p(y | x_t)` 采样 y。

下游（双流 `.bin` 写盘、`meta.pkl`、trainer/Hessian 的 `get_batch`）都已经支持，
无需再改动其它代码。
