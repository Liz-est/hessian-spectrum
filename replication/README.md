# replication/ — RotatedMatrixBigramProblem 复现

独立于 `toy_models/` 流程的最小复现：完全按照对方代码的解析式 population 目标
（不采样数据、不用 Transformer 训练框架）。

## 问题定义（problem.py，逐行移植自对方代码）

- `E`：固定 dim×dim **正交矩阵**（seed=dim 的高斯 QR），即 frozen embedding。
- `W`：dim×dim 可训练参数（lm_head），**零初始化**，无 bias。
- `pi`：精确 Zipf(s=1) 权重，`pi_i ∝ 1/i`（解析权重，非采样经验分布）。
- loss：`f(W) = Σ_i pi_i · 0.5‖e_iᵀEW − e_iᵀ‖²`，即 identity 任务（y==x）的
  0.5·**per-class-sum** 平方误差（不是 mean）。
- 梯度用对方的解析 `full_grad`（已对 autograd 校验到 1e-17）。

性质：Hessian 恒定 = `(Eᵀdiag(pi)E) ⊗ I`，特征值**精确等于 {pi_i}**（各重数
dim）。dim=10000 时 λ_max=pi_1≈0.102，λ_min=pi_10000≈1.02e-6，条件数 = 10000。
GD 稳定上界 lr < 2/λ_max ≈ 19.6。W=0 处 loss = 0.5。

## 与 toy_models 旧 REP-* 实验的关键差异（复现失败的原因）

1. embedding 是精确正交阵（行范数 1），不是 N(0,0.2²)（行范数≈20）；
2. loss 是 0.5·per-class-sum，比 `F.mse_loss` mean 大 V/2=5000 倍 → lr 不可比；
3. 按解析 pi 精确加权的确定性全梯度，无 100k-token 采样噪声（尾部 class 不缺失）；
4. W 无 bias。

## 两种梯度模式（每个 optimizer×lr 组合各跑一份）

- **population**（对方的原始设置）：确定性 `full_grad`，精确 pi 加权 —— 等价
  于无限数据。run 名无后缀。
- **stochastic**：每步新采 batch_size=100k 个 i.i.d. class x~pi（identity 数据
  y==x），梯度用 batch 的经验频率加权（`weighted_grad`，已对 autograd 校验；
  因为 per-sample loss 只经 class 进入，采 batch == 采 Multinomial counts）。
  run 名带 `-stoch` 后缀。**评估口径两种模式统一用 population loss**，可直接
  对比曲线。

## 文件

- `problem.py` — 问题定义（对方代码 + weighted_grad / hessian_eigs 便利函数）
- `presets.py` — RunConfig + lr 网格（adam 1e-2…3e-4；sgd 1…15）×
  {population, stochastic}，共 16 个正式 preset；对方未给 optimizer 设置，
  adam betas=(0.0,0.999) 沿用旧 REP-* 约定，lr 按本 loss 归一化重标
- `train.py` — 单卡训练；输出 loss_log.csv（含 10 个等 pi 质量频率组的分组
  loss）、11 个 ckpt 的 per-class loss 向量、两张图（标题/图例含完整优化器设置）
- `plot_compare.py` — 所有 run 汇总对比图（Adam 蓝 / SGD 橙，population 实线 /
  stochastic 虚线，图例含完整优化器设置）→ runs/compare_loss.png、
  runs/compare_groups.png
- `run_all.py` — 一卡一 preset 并行跑全部（16 run 分两波占满 8×H100）
- `submit_sco.py` — SCO 提交（平台参数同 toy_models/submit_sco_vanilla.py），
  job 内 run_all.py 后自动跑 plot_compare.py

## 用法

```bash
# 本地冒烟（dim=256, CPU）
python train.py rep-identity-smoke --device=cpu
python train.py rep-identity-smoke-stoch --device=cpu
# 单个正式 run
python train.py rep-identity-adam-lr1e-3          # population
python train.py rep-identity-adam-lr1e-3-stoch    # stochastic
# 全部并行 + 汇总图
python run_all.py            # 可加子串过滤：python run_all.py sgd / stoch
python plot_compare.py
# 提交 SCO（16 run + 汇总图，一个 job）
python3 submit_sco.py --yes
```

输出在 `replication/runs/<preset>/`。
