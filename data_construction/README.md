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

## 数据构造的数学过程（详细）

记词表大小为 $V$，token 取值于 $\{0, 1, \dots, V-1\}$。数据流是一条一阶马尔可夫链
$x_1, x_2, \dots$，完全由二元组 $(\pi, P)$ 刻画：$\pi \in \Delta^{V-1}$ 是初始/平稳分布，
$P \in \mathbb{R}^{V \times V}$ 是行随机的转移矩阵（$P_{ij} = \Pr[x_{t+1} = j \mid x_t = i]$）。
构造分六步。

### 第 1 步：目标词频分布 π（`make_pi`）

按 `freq` 取未归一化权重 $w$：

$$
w_i =
\begin{cases}
1 & \text{uniform（balance）} \\
(i+1)^{-s} & \text{zipf，指数 } s = \texttt{zipf\_s} \text{（imbalance）} \\
c_i & \text{real，} c \text{ 为真实语料的 unigram 计数}
\end{cases}
$$

再做平滑与归一化：$\pi_i = (w_i + \varepsilon) / \sum_j (w_j + \varepsilon)$，$\varepsilon = 10^{-12}$。
平滑保证 $\pi$ **严格正**，从而后面 MH 接受率里的比值 $\pi_j / \pi_i$ 与 $\log \pi$ 处处有限。
Zipf 情形下 $s = 0$ 退化为 uniform，$s = 1$ 是经典 Zipf，$s > 1$ 更重尾——`zipf_s`
就是"不均衡程度"的连续旋钮。

### 第 2 步：集中的提议核 Q（`make_proposal`）

在 token 索引上定义**环形距离** $d(i, j) = \min(|i - j|,\ V - |i - j|)$，带宽
$h = \max(\texttt{bandwidth\_frac} \cdot V,\ 10^{-6})$，构造高斯 bump 并按行归一化：

$$
Q_{ij} = \frac{\exp\!\big(-d(i,j)^2 / 2h^2\big)}{\sum_k \exp\!\big(-d(i,k)^2 / 2h^2\big)}
$$

两个关键性质：

1. **对称性**：$d(i,j) = d(j,i)$，且由环形平移不变性每行的归一化常数相同，故
   $Q_{ij} = Q_{ji}$ 精确成立。
2. **低熵**：`bandwidth_frac` 小 → 每行质量集中在索引相邻的少数 token 上 →
   提议（进而 B）尖锐、可学习。

token 的索引顺序本身是任意的（token 没有天然顺序）；环形结构只是构造一个良定义、
可复现、低熵核的手段。

### 第 3 步：Metropolis–Hastings 核 B（`make_mh_kernel`）

以 $\pi$ 为目标分布、$Q$ 为提议，接受率与核为

$$
\alpha_{ij} = \min\!\Big(1,\ \frac{\pi_j Q_{ji}}{\pi_i Q_{ij}}\Big), \qquad
B_{ij} = Q_{ij}\,\alpha_{ij} \ (j \neq i), \qquad
B_{ii} = 1 - \sum_{j \neq i} B_{ij}
$$

被拒绝的提议质量全部留在对角线上（自环）。由于本文的 $Q$ 对称，接受率化简为
$\alpha_{ij} = \min(1, \pi_j / \pi_i)$：向高频 token 的转移总被接受，向低频 token 的
转移按频率比概率接受——这正是 $\pi$ 得以成为平稳分布的机制。

**细致平衡（detailed balance）证明**：对 $j \neq i$，

$$
\pi_i B_{ij}
= \pi_i Q_{ij} \min\!\Big(1, \frac{\pi_j Q_{ji}}{\pi_i Q_{ij}}\Big)
= \min\big(\pi_i Q_{ij},\ \pi_j Q_{ji}\big)
$$

右端关于 $(i, j)$ 对称，故 $\pi_i B_{ij} = \pi_j B_{ji}$。两边对 $i$ 求和：

$$
(\pi B)_j = \sum_i \pi_i B_{ij} = \sum_i \pi_j B_{ji} = \pi_j \sum_i B_{ji} = \pi_j
$$

即 $\pi B = \pi$——**对任意提议 Q 都精确成立**，这是 MH 构造的核心保证。

### 第 4 步：混合转移矩阵 P（`build_transition`）

定义秩 1 的"独立核" $\Pi_{\text{indep}} = \mathbf{1}\pi^{\top}$（每行都等于 $\pi$）。
它显然平稳：$(\pi \Pi_{\text{indep}})_j = \sum_i \pi_i \pi_j = \pi_j$。最终转移矩阵取凸组合

$$
P = (1 - a)\,\Pi_{\text{indep}} + a\,B, \qquad a = \texttt{predictability} \in [0, 1]
$$

**平稳性对任意 a 成立**（这就是词频与难度解耦的全部证明）：

$$
\pi P = (1 - a)\,\pi \Pi_{\text{indep}} + a\,\pi B = (1 - a)\,\pi + a\,\pi = \pi
$$

采样上的直观解释：每一步以概率 $1 - a$ **无视上文**、直接从边际 $\pi$ 抽下一个 token；
以概率 $a$ 从尖锐核 $B$ 抽。$a$ 越大，上文越"有用"，任务越可学。

代码最后做 `clip` + 行归一化只是清理浮点负零/漂移（幅度在机器精度量级），实测对平稳性
的扰动为 $\mathrm{TV}(\hat\pi, \pi) \approx 10^{-10}$。

### 难度的信息论刻画（predictability 到底控制什么）

对完美拟合了 $P$ 的模型，next-token 交叉熵的下界（Bayes 风险）就是链的**条件熵率**：

$$
H(x_{t+1} \mid x_t) = \sum_i \pi_i\, H(P_{i\cdot}), \qquad
H(P_{i\cdot}) = -\sum_j P_{ij} \log P_{ij}
$$

- $a = 0$：$P_{i\cdot} = \pi$，条件熵 $= H(\pi)$（该约束下的最大值）。上文完全无信息，
  模型最多学会 unigram 词频，最优 loss $= H(\pi)$。
- $a = 1$：$P = B$，条件熵 $\approx$ B 的行熵，由 `bandwidth_frac` 控制（带宽小 → 熵低 →
  最优 loss 低）。
- 中间 $a$：每行是 $\pi$ 与 $B_{i\cdot}$ 的线性插值，熵在两端点值之间连续过渡（由熵的凹性，
  中间值不低于两端点熵的线性插值）。

所以 `predictability` 控制的是"最优 loss 能压到多低"，而 `bandwidth_frac` 是次级旋钮，
控制"最易端（$a=1$）有多易"。`inspect_dataset.py` 画的每行熵直方图直接可视化这个量。

### 第 5 步：序列采样（`sample_sequence`）

$$
x_1 \sim \pi, \qquad x_{t+1} \mid x_t \sim P_{x_t \cdot}
$$

因为初始 token 直接取自平稳分布，链**处处平稳**（无需 burn-in）：对每个 $t$ 都有
$\Pr[x_t = i] = \pi_i$ 精确成立。实现上预计算每行的 CDF，用一批均匀随机数
$u_t \sim U(0,1)$ 做逆变换采样（`searchsorted`）。

流按 `seq_len` 分块独立采样后拼接：每块各自平稳，因此拼接流的 unigram 频率仍精确是
$\pi$；代价是每隔 `seq_len` 个 token 有一处"接缝"转移不服从 $P$（占比
$1/\texttt{seq\_len} \approx 0.1\%$，对边际分布无影响）。

### 第 6 步：标签构造（`build_targets`）

`shift` 模式（默认）：$y_t = x_{t+s}$（$s = \texttt{shift}$，默认 1），即标准
next-token prediction。实现上 x 截掉尾部 $s$ 个 token、y 截掉头部 $s$ 个，与原
NanoGPT 单流格式在数值上完全一致。由平稳性，y 的边际分布同样是 $\pi$，且
$x_t \to y_t$ 的条件分布为 $P^s$（$s$ 步转移矩阵）。

### 生成后的数学校验

| 量 | 定义 | 期望值 |
|----|------|--------|
| 平稳性检查 | $\mathrm{TV}(\hat\pi, \pi) = \tfrac12 \lVert \hat\pi - \pi \rVert_1$，$\hat\pi$ 为幂迭代求得的 $P$ 的不动点 | $\approx 10^{-10}$ |
| 经验词频 | 同上，但 $\hat\pi$ 为写盘 token 流的经验频率 | $O(\sqrt{V/N})$ 的采样噪声 |
| 每行熵 | $H(P_{i\cdot})$ 的直方图 | 随 `predictability` 从 $\log V$ 附近移向低熵端 |

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
