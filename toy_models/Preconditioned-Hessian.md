# Preconditioned-Hessian Block-Heterogeneity Metric

本文档记录"预条件 Hessian 的 block 间/block 内异质性二维图"的**完整计算流程**，逐条标明用哪个文件、哪个函数、哪几行实现，以及**每一步是否存在近似**。

分析对象：0 层 toy 模型（`n_layer=0`，仅 `tok_emb` + `lm_head` 两个权重矩阵），`loss_type="mse_rep"`，词表 `V = n_embd = d = 10000`。三个优化器 SGD / Adam / Muon，各自训练出一组 checkpoint（init, p10, …, p100）。

核心动机：**原始 Hessian $H$ 与优化器无关**（同一模型同一数据，$H$ 一样），三条轨迹会重合、看不出区别。真正区分优化器的是**预条件后的 Hessian $P^{-1}H$**（$P$ = 该优化器的预条件子），这才是各优化器在参数空间实际"感受"到的曲率。

主实现文件：[`compute_precond_hessian.py`](compute_precond_hessian.py)
绘图文件：[`plot_precond_2d.py`](plot_precond_2d.py)（谱熵版 Y 轴）、[`plot_precond_2d_stdlog.py`](plot_precond_2d_stdlog.py)（std-log10 版 Y 轴）

---

## 0. 记号

- $d$ = 隐藏维 = 10000；$C = V$ = 词表 = 10000。
- 被分析层的权重矩阵按**行**分块：每一行是一个 block。
  - `lm_head`：block $k$ = 输出 class $k$ 的那一行（共 $C$ 个 block）。
  - `embedding`：block $v$ = token $v$ 的 embedding 行（共 $V$ 个 block）。
- $H_k$ = 第 $k$ 个 block 的 $d\times d$ Hessian 子块（block-diagonal 的对角块）。
- 实际只分析前 `max_blocks`（默认 256）个 block。

---

## 1. Hessian 是如何得到的（是否近似）

关键结构（本项目能高效计算的根本原因）：**每个 block 的原始 Hessian 都能写成"一个共享矩阵 $M$ × 一个逐 block 标量 $\text{scale}_k$"**：
$$H_k = \text{scale}_k \cdot M$$

实现：[`compute_precond_hessian.py`](compute_precond_hessian.py) 的 `block_hessian_factor()`（**L175–227**）。

### 1a. lm_head 层（L182–205）

数学形式：
$$H_k = c\cdot \frac{1}{N}\sum_{t} x_t x_t^\top,\qquad \text{scale}_k \equiv 1$$
其中 $x_t$ 是 lm_head 的**输入激活**（即最后一层隐藏态），$N$ = 总 token 数，$c$ 由 loss 约定决定：`mse_rep` 下 $c=1$，`mse` 下 $c=2/C$（L186）。

- $x_t$ 通过 **forward hook** 抓取（L194：`register_forward_hook`，`cap["x"]=i[0]`，即 module 的输入）。
- Gram 矩阵 $\sum_t x_t x_t^\top$ 用 `feat.t() @ feat` 累加（L200），最后 $M = c\cdot(\text{Gram}/N)$（L203）。
- `scales = ones(C)`（L204）——**所有 class 的 block 完全相同**（`mse_rep` 曲率与 class 无关）。

**是否近似**：**精确**（对 mse/mse_rep 是 loss 对 logits 的精确二阶导 = $I_C\otimes(c\,\tfrac1N\sum x x^\top)$，无 Gauss–Newton 残差项，因为 MSE 对 logits 是二次型）。⚠️ 一个**采样近似**：$\frac1N\sum_t x_t x_t^\top$ 只在**一个 full-batch**（`get_batch` 取数据集前 N 个 token，见 L103–122）上估计，不是整个数据集。对 `mse_rep` 而言 $x_t$ 只依赖输入 token（0 层无位置编码），full-batch 已覆盖足够 token，误差可忽略。
CE loss 未实现（L187–190 直接报错），因为 CE 的 block 依赖 class（$p_k(1-p_k)$ 加权），不存在共享 $M$。

### 1b. embedding 层（L207–225）

数学形式（0 层 Gauss–Newton，**精确**）：
$$H_v = \frac{N_v}{N}\, W^\top W,\qquad M = W^\top W,\quad \text{scale}_v = \frac{N_v}{N}$$
其中 $W$ = `lm_head.weight`（$C\times d$），$N_v$ = token $v$ 在 batch 中出现次数。

- $M = W^\top W$ 直接矩阵乘（L211）。
- $\text{scale}_v = N_v/N$ 由 forward hook 抓 embedding 的输入 id、`index_add_` 计数得到（L216–224）。

**是否近似**：**精确**，但**仅当 `n_layer==0`**（L208 有 `assert`）。原因：0 层时 logits $z = W e_v$ 对 embedding 行 $e_v$ 是**线性**的，故精确 Hessian = Gauss–Newton block，残差项为零。同样有 1a 的 full-batch 采样近似（$N_v/N$ 是经验词频）。

> 备注：这与 [`hessian_toy.py`](hessian_toy.py) 里 `last_layer_blocks`（L200+）/`token_embedding_gn_blocks`（L330+）的 raw-Hessian 计算约定一致，本脚本重算 $M$ 而非复用，以便接预条件子。

---

## 2. 三种优化器的预条件 Hessian（是否近似）

预条件 Hessian 取**对称形** $\tilde H_k = P_k^{-1/2} H_k P_k^{-1/2}$（它与非对称的 $P_k^{-1}H_k$ **谱相同**——相似变换，但对称形保证实特征值）。主循环在 [`compute_precond_hessian.py`](compute_precond_hessian.py) `main()` **L340–377**。

### 2a. SGD（L340–345）

$P = I$，故 $\tilde H_k = H_k = \text{scale}_k\cdot M$。
- 只对 $M$ 做一次特征分解 `eM = eigvalsh(M)`（L342）。
- **无近似**（除 1 的 full-batch 采样）。所有 block 是 $M$ 的同一套谱乘 $\text{scale}_k$。

### 2b. Muon（L347–354）

来自 [`config/build.py`](config/build.py) 的真实实现：更新方向是 $\text{NS}_5(G)$，其中 $\text{NS}_5$ 是 5 步 Newton–Schulz 正交化（`_zeropower_via_newtonschulz5`, build.py L32–50）。理想上 $\text{NS}_5(G)\to \text{msign}(G)=UV^\top$（$G=USV^\top$），对应右预条件子
$$P = (G^\top G)^{1/2}.$$

实现 `muon_Pinv_sqrt()`（**L255–267**），用一个**NS5-精确恒等式**而非理想公式：
$$\text{msign}(G)^\top G = (VU^\top)(USV^\top) = VSV^\top = (G^\top G)^{1/2} = P$$
- L262：`msign = _zeropower_via_newtonschulz5(G, steps=ns_steps)`（NS5 代码复制自 build.py，系数 `3.4445, -4.7750, 2.0315`，L70）。
- L263：`P = msign.t() @ G`（= NS5 版的 $(G^\top G)^{1/2}$）。
- L264：`P = 0.5*(P+P.t())` 对称化（消除 NS5 数值不对称）。
- L265–267：`eigh` 得 $P^{-1/2}$。
- L349–350：`Mpc = Pis @ M @ Pis`，再 `eigvalsh`（L351）。

**是否近似**：**有近似，且刻意保留 NS5 的近似**。用 `msign(G)^T @ G` 而非直接算 $(G^\top G)^{1/2}$，是为了让分析**忠实反映优化器真实用的 NS5 近似**（而不是理想极分解）。NS5 本身：5 步迭代、bfloat16 运算（build.py L40）、初始按 Frobenius 范数归一化——这些近似都被包含进来。另有对称化（L264，消除微小不对称）和 PD 下界 `clamp_min(1e-12)`（L266，数值保护）。
$G$ 是 full-batch 梯度（见 §2 尾）。因 embedding 冻结/共享，$P$ 对所有 block **相同** → Muon 的所有 block 仍全同。

### 2c. Adam（L356–377）

来自 [`config/build.py`](config/build.py) 的配置：`betas=(0.0, 0.999)`，即 **$\beta_1=0$**（无一阶动量），更新 $=-\eta\, g/\sqrt{\hat v}$，$\hat v$ = 二阶矩 EMA。故预条件子是**逐参数对角**、**逐 block 不同**：
$$P_k = \mathrm{diag}\!\big(\sqrt{\hat v_k}\big),\qquad D_k := P_k^{-1/2} = \mathrm{diag}\!\big(\hat v_k^{-1/4}\big)$$
$$\tilde H_k = D_k\,H_k\,D_k = \text{scale}_k\cdot D_k M D_k$$

- **二阶矩 $\hat v$ 的获取**（`adam_replay_ema`, **L233–252**）：checkpoint **没有保存优化器状态**，故从 init 开始**精确重放** EMA：`v = β₂·v + (1-β₂)·g²`（L248），带 bias-correction `v/(1-β₂^t)`（L249）。$g$ 是每个 checkpoint 的 full-batch 梯度。
- **对角因子**：`p = 1.0/(v_hat**0.25 + EPS)`（**L361**）—— 指数 **$-1/4$** 对应对称形 $P^{-1/2}=\mathrm{diag}(\hat v^{-1/4})$。
  > ⚠️ 历史 bug：曾写成 `1/sqrt(v_hat)`（即 $\hat v^{-1/2}$，等价于用了 $P=\mathrm{diag}(\hat v)$），2026-08-06 修正为 $\hat v^{-1/4}$。修正后 X 轴数值约减半，定性结论不变。
- $\tilde H_k = D_k M D_k$：L371 `Hk = pk[:,None]*M*pk[None,:]`。

**是否近似**：
1. **$\hat v \approx$ 真实 Adam 的 $v$**：因 checkpoint 无优化器态，重放 EMA。理论上精确（用了真实 $\beta_2$、真实每步梯度）—— 但**这里每个"tag 间隔"只算一次梯度**（在 checkpoint 处），而真实训练在两个 checkpoint 之间跑了很多 step，梯度在变。所以这是**"每个 checkpoint 采一次梯度的 EMA"近似**，非逐 step 重放。因是 full-batch 训练、接近不动点时 $v_t\to g^{\odot2}$，该近似合理但**非严格**。
2. **对角近似本身不是近似**：$\beta_1=0$ 下 Adam 预条件子确实是对角，无省略。
3. full-batch 梯度采样（同上）。

### 梯度 $G$ 的计算（供 2b/2c）

`compute_gradient_fullbatch()`（**L131–159**）：对被分析层的权重做一次 full-batch 前向+反向，取 `param.grad`（L157）。
- L137 `param.requires_grad_(True)`：即使该层训练时被冻结，分析时也强制打开梯度（否则 `.grad` 为 None）。
- `chunk_seqs` 选项（L144–151）：显存紧张时按序列分块累加梯度（loss 是对位置取 mean，故每块按 kept-token 占比加权，L151），保证 = 单次 full-batch 梯度。H100 上不需要。

---

## 3. 预条件 Hessian 的 block 间 / block 内特征值（是否近似）

特征分解统一用 `eigvalsh()`（**L285–286**，`torch.linalg.eigvalsh`，对称矩阵实特征值，升序），fp64。

### 3a. block 内特征值（within-block，用于 Y 轴）

即单个 block 的 $d=10000$ 个特征值 $\{\lambda_i^{(k)}\}$。

- **SGD**（L342）：所有 block 谱相同，只算一次 `eigvalsh(M)`，存 1 条谱。**精确**。
- **Muon**（L351）：所有 block 谱相同（共享 $P$），只算一次 `eigvalsh(Mpc)`，存 1 条谱。近似来自 §2b 的 NS5。
- **Adam**（L366–377）：每个 block 的 $D_k$ 不同，谱**逐 block 不同**，需对每个 block 做 `eigh`。$256\times$`eigh(10000²)` 太贵，故**只对前 `subsample_eigh`（默认 16）个 block 做完整 `eigh`**（L369–375），Y 轴对这 16 个求平均。
  > **这是一个采样近似**：Y_within（Adam）= 前 16 个 block 的谱熵均值，不是全 256 个。由于谱熵尺度不变、block 间谱形状相近，16 个的均值是合理估计，但非全量。

### 3b. block 间特征值（between-block，用于 X 轴）

用每个 block 的**平均特征值** $\bar\lambda_k = \frac1d\sum_i \lambda_i^{(k)} = \frac1d\mathrm{tr}(\tilde H_k)$（一个标量/ block）。

- **SGD/Muon**（L343 / L352）：$\bar\lambda_k = \text{scale}_k\cdot(\text{eM 或 eMpc 的均值})$。因共享谱，between 差异**纯粹来自 $\text{scale}_k$**。
- **Adam**（L362–364）：用 trace 恒等式**避免 eigh**：$\bar\lambda_k = \text{scale}_k\cdot\frac1d\sum_i p_{k,i}^2\,H_{ii}$（L364，$H_{ii}$ = `Hdiag`，$p_{k,i}=\hat v_{k,i}^{-1/4}$）。因 $\mathrm{tr}(D_k M D_k)=\sum_i p_{k,i}^2 M_{ii}$，这是**精确的**平均特征值，且**用到全部 256 个 block**（不采样）。

存盘：`block_mean.npy`（全 block 的 $\bar\lambda_k$）+ `eigs.npy`（within 谱：SGD/Muon 1 条，Adam 16 条），L382–383。

---

## 4. 两个 metric（对 block 间/内特征值做什么计算）

### 4a. X 轴 = between-block 异质性（scale 上的差异）

$$X = \mathrm{std}_k\big(\log_{10}\bar\lambda_k\big)$$

- 实现：`main()` **L379–380**：`pos = block_mean[block_mean>0]; X = std(log10(pos))`。
- 含义：各 block 平均曲率跨越几个数量级。**尺度敏感**、$0$ = 所有 block 一致、越大越异质。
- 只用正的 $\bar\lambda_k$（L379 过滤）。
- **SGD/Muon 恒为 0**（所有 block 共享谱，$\bar\lambda_k$ 只差一个 $\text{scale}_k$；但注意：若 $\text{scale}_k$ 本身有差异——如 embedding 的词频——X 仍非零）。**Adam 是唯一因逐 block 预条件而在 lm_head 上产生 X>0 的优化器**。

为何 between **不**用谱熵：between 的对象是 256 个**标量**（不是一个谱），且需**对 scale 敏感**（谱熵会先归一化、抹掉绝对尺度）。std(log) 直接读出"跨几个数量级"，方向直觉也对（大=异质）。

### 4b. Y 轴 = within-block 异质性（单 block 谱的不均匀度）

有**两个版本**：

**版本 A — 谱熵**（默认，`plot_precond_2d.py`）：
$$Y = \operatorname*{mean}_{k}\Big[ -\sum_i \tilde\lambda_i^{(k)}\log\tilde\lambda_i^{(k)}\;/\;\log d \Big],\qquad \tilde\lambda_i = \frac{\lambda_i}{\sum_j\lambda_j}$$
- 实现：`spectral_entropy()` **L273–282**。先 `clip(0)`（丢负特征值，L274），归一化成概率（L278），丢 $\tilde\lambda<10^{-15}$（L279，避免 $0\log0$ 且 $\log$ 分母用有效个数），香农熵 $/\log(\text{有效个数})$（L282）。
- 含义：**尺度不变**（block 乘任意常数不变），$1$ = 谱完全均匀（所有特征值相等），$\to 0$ = 极不均匀。**越大 = 越均匀**。
- Adam 对 16 个 subsample block 求均值（L374、L376）；SGD/Muon 单条谱（L344/L353）。

**版本 B — std(log10)**（`plot_precond_2d_stdlog.py`，从已存 eigs 重算，不重跑）：
$$Y = \operatorname*{mean}_{k}\Big[ \mathrm{std}_i\big(\log_{10}\lambda_i^{(k)}\big) \Big]$$
- 实现：`plot_precond_2d_stdlog.py` 的 `std_log10_within()`（**L74–88**）。
- **含义与谱熵方向相反**：越大 = 谱跨越越多数量级 = **越不均匀**。使两个轴统一为"右上=最异质"。
- **有效谱截断（关键近似）**：谱里约 7% 是数值零（谱有个 $10^{-4}\to10^{-14}$ 的断崖），log10 对它们极敏感。故只取 $\lambda > \lambda_{\max}\cdot 10^{-6}$ 的有效特征值再算 std（L83：`keep = pos[pos > pos.max()*REL_FLOOR]`，`REL_FLOOR=1e-6`，L38）。谱熵天然对这些零鲁棒，std(log10) 则**必须人为截断**——这是 std-log 版相对谱熵的固有代价。

### 两 metric 对比

| | X (between) | Y-A 谱熵 (within) | Y-B std-log10 (within) |
|---|---|---|---|
| 对象 | 256 个 block 均值(标量) | 单 block 的 d 个特征值 | 单 block 的 d 个特征值 |
| 尺度 | 敏感(要看 scale) | 不变(归一化掉) | 不变(取 std) |
| 方向 | 大=异质 | 大=均匀 | 大=异质 |
| 数值零 | 只过滤 ≤0 | 天然鲁棒 | **需截断 λ<λmax·1e-6** |
| 近似 | Adam 用 trace 全量精确 | Adam 采样 16 block | Adam 采样 16 block + 截断 |

---

## 5. 近似汇总（一览）

| 步骤 | 近似 | 位置 |
|---|---|---|
| raw Hessian $M$ | full-batch 采样估 $\frac1N\sum xx^\top$ / $N_v/N$ | L196–203, L218–224 |
| lm_head 精确性 | 精确(mse/mse_rep 无 GN 残差)；CE 不支持 | L186–190 |
| embedding 精确性 | 精确**仅当 n_layer==0**（线性→GN 精确） | L208 assert |
| Muon $P$ | **刻意保留 NS5 近似**(5 步/bf16/归一化) + 对称化 + PD 下界 | L255–267, build.py L32–50 |
| Adam $\hat v$ | 每 checkpoint 采一次梯度的 EMA 重放(非逐 step) | L233–252 |
| Adam 预条件对角 | 精确(β₁=0 下确为对角) | L361 |
| within 谱(Adam) | 只对前 16 个 block 做 eigh，Y 取均值 | L366–377 |
| between 均值(Adam) | trace 恒等式，**全量精确** | L362–364 |
| X = std(log10 mean) | 只过滤 ≤0 | L379–380 |
| Y 谱熵 | 丢 λ̃<1e-15 | L273–282 |
| Y std-log10 | 截断 λ<λmax·1e-6 | stdlog L74–88 |
| 梯度 $G$ | full-batch 一次前向反向 | L131–159 |

---

## 6. 运行

```bash
# 计算（需 GPU / SCO H100，d=10000 本地 8GB 会 OOM）
python compute_precond_hessian.py runs/<run> --layer {lm_head|embedding} --optim {sgd|adam|muon} --device cuda
# 提交 SCO（按 group）
python submit_sco_precond.py --group {frz_embd|frz_lmhead|all} --yes
# 画图：谱熵版 Y
python plot_precond_2d.py runs/<sgd> runs/<adam> runs/<muon> --out_dir files/precond_2d_<group>
# 画图：std-log10 版 Y（复用已存 eigs，不重跑）
python plot_precond_2d_stdlog.py --group {frz_embd|frz_lmhead|all}
```

输出：`runs/<run>/precond_hessian/<layer>_<optim>/{<tag>_eigs.npy, <tag>_block_mean.npy, <tag>_summary.json, all_summary.json}`；图在 `files/precond_2d_<group>[_stdlog]/precond_2d_{lm_head,embedding}.png`。
