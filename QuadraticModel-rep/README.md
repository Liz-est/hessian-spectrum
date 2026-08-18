# QuadraticModel-rep

PyTorch 复现 "A Defense of the Quadratic Model" (arXiv 2607.21716) Figure 2 的 Hessian 谱。
原论文 JAX 仓库见 `../QuadraticModel/`（只有预计算缓存 + 绘图，无谱生成代码与 checkpoint）。

## 目录结构（2026-08 重构）

```
src/            功能代码（package，from src.xxx import ...）
  paths.py        所有数据/产物路径的唯一来源（见下）
  config.py       RunConfig + PRESETS + build_optimizer；python src/config.py 自检
  model/          model.py(Transformer) layers.py(层划分) blocks.py(unit 子块)
  data/           fineweb_data.py data_grain.py(grain parquet 管线) tokenize_*.py
  optim/          opt.py(CompletePAdam/CompletePMuon/EMA)
  train/          train.py（torchrun 入口）
  spectrum/       hvp/lanczos/gauss_radau 核心算子 + spectrum_ddp 等谱计算入口
  analysis/       verify_*/analyze_blocks/dense_block_eig 等验证与逐块分析
sco/            SCO 集群提交脚本（submit_* 提交，run_*_inner 在 worker 上执行）
scripts/        一次性工具与绘图（plot_*、compare_with_cache、reband_all 等）
tokenizers/     bpe_8192.json（与论文同一份）
```

## 路径约定（src/paths.py）

- **只读共享资源指主目录** `/data/250010020/hessian-spectrum/`：
  100BT parquet 数据集（`HS_DATA_DIR` 覆盖）、论文谱缓存 spectrum_3x3.npz（`HS_CACHE_NPZ` 覆盖）。
- **产物全在本仓库根下**：checkpoints_b64*/、outputs/、figures/、test_outputs/（gitignore，重训自产）。
- 代码用 `__file__` 自适应仓库根：主目录 clone 跑 = 主目录实验，worktree 分支跑 = 分支实验，脚本不用改。
- 对比合作者结果：`HS_PEER_REP` 指对方 QuadraticModel-rep（默认主目录那份）。

## 常用命令

```bash
# 训练（8 卡）
torchrun --standalone --nproc_per_node=8 src/train/train.py b64_adam
# 谱计算（DDP）
torchrun --standalone --nproc_per_node=8 -- src/spectrum/spectrum_ddp.py --ckpt checkpoints_b64/ckpt_p100.pt --m 1200 --out outputs/xxx.npz
# SCO 提交
bash sco/submit_train_b64.sh
# 路径自检
python src/paths.py && python src/config.py
```

⚠️ 文件名/npz tag 里的 `sgd`（gn_sgd/hessian_sgd）实际含义是 **raw = CompleteP 预条件 √(pre·post)**，
非真 SGD；为对齐论文缓存的 key 故意不改。
