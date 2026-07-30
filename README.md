# Data Imbalance & Hessian Heterogeneity (bigram-senmiao)

研究 **数据不均衡（词频分布）× 优化器（AdamW vs SGD/GD）× 模型结构** 对训练动态
和 Hessian 谱/异质性的影响。基于论文 *Why Transformers Need Adam: A Hessian
Perspective*（[arXiv:2402.16788](https://arxiv.org/abs/2402.16788)）的代码库
扩展：用可控合成 bigram 数据 + 小型 Transformer 做干净的受控实验。

## 目录结构

```
data_construction/       合成 bigram 数据构造（词频 π / 可预测性 / 标签模式三个
                         正交旋钮；另有只控边际的 shuffled 构造）。详见其 README
toy_models/              核心实验：小型 decoder-only Transformer 的训练 +
                         两条 Hessian 分析路线（逐单元精确块谱 / 全参数 SLQ 谱）。
                         详见其 README
language_models/test_data/   tiny GPT-2 对照实验的独立副本（train_gpt2.py +
                         hessian_spectrum.py，已改造支持双流合成数据）
linear_bigram_mse_gradient_hessian.{tex,pdf}   线性 bigram 模型 MSE 损失的
                         梯度/Hessian 理论推导（shuffled 数据构造的动机）
```

## 典型工作流

```bash
# 1. 生成合成数据（balance vs imbalance 对照）
cd data_construction
python build_dataset.py configs/uniform_balanced.py
python build_dataset.py configs/zipf_imbalanced.py

# 2. 训练 + Hessian 分析（preset 驱动，见 toy_models/config/presets.py）
cd ../toy_models
torchrun --standalone --nproc_per_node=8 train_vanilla_transformer.py layer5-imbalance-s1-adamw
python3 analyze_vanilla.py layer5-imbalance-s1-adamw        # 逐单元块谱
python3 analyze_full_spectrum.py layer5-imbalance-s1-adamw  # 全参数 SLQ 谱

# 或提交 SCO 单 job 一体化跑完（训练 -> 块谱 [-> SLQ]）
python3 submit_sco_vanilla.py
```

## 上游与致谢

代码基于 [hessian-spectrum](https://github.com/zyushun/hessian-spectrum)
（原论文实现，SLQ 谱估计）与 [NanoGPT](https://github.com/karpathy/nanoGPT/)。
原 repo 的 vision_models 与完整 language_models 线与本分支方向无关，已移除
（可在 git 历史或 `main` 分支找回）。

注意：Hessian-vector product 不支持 Flash Attention——谱估计相关代码里 attention
必须走朴素实现（`sdpa_kernel(SDPBackend.MATH)`），且 forward/backward 需去除一切
随机性（数据 shuffle、dropout）。

原论文引用：

```
@article{zhang2024why,
  title     = {Why Transformers Need Adam: A Hessian Perspective},
  author    = {Zhang, Yushun and Congliang, Chen and Tian, Ding and Ziniu, Li and Sun, Ruoyu and Luo, Zhi-Quan},
  booktitle = {arXiv preprint arXiv:2402.16788},
  year      = {2024},
}
```
