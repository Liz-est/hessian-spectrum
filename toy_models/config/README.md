# Toy Model 配置说明

所有实验设置(模型结构、数据、优化器及其超参、学习率调度、训练循环、checkpoint
计划、Hessian 分析参数)都集中在 `config/` 包,不再散落在各脚本的模块级全局变量里。
train / analyze 脚本从同一份 `ExperimentConfig` 读取,保证两阶段的 `run_name`、
checkpoint 计划、模型结构不会漂移。

## 文件列表

* `config/schema.py` — 分组 dataclass:`ModelConfig` / `DataConfig` / `OptimConfig` /
  `LRConfig` / `TrainConfig` / `AnalyzeConfig`,组成顶层 `ExperimentConfig`。
  `ckpt_iters()` 带碰撞检测(两个 fraction 舍入到同一 iter 会报错);
  `to_model_config()` 把 `ModelConfig` 转成模型类要的 `ToyVanillaConfig`。
* `config/build.py` — `build_optimizer()`(按 `optim.name` 分发 sgd/adamw/adam)和
  `make_lr_fn()`(warmup + cosine/constant)。
* `config/presets.py` — 当前实验的命名预设字典 `EXPERIMENTS`。
* `config/legacy_presets.py` — 历史实验归档，不会被 `config.load()` 自动注册。
* `config/__init__.py` — `load(name)` 返回预设的深拷贝;`apply_overrides(cfg, argv)`
  应用 CLI 覆盖。

## 当前预设（`config/presets.py`）

| 预设名 | 模型 | 优化器 | run_name / files_name |
|---|---|---|---|
| `fullbatch-mse0-shuffled-2p17-gd` | 无位置编码的 embed→lm_head，MSE，n_layer=0 | full-batch GD | 同 preset 名 |
| `fullbatch-mse0-shuffled-2p17-adam` | 无位置编码的 embed→lm_head，MSE，n_layer=0 | full-batch Adam | 同 preset 名 |

两项只改变优化器及其学习率。共同设置包括：`2^17` 个训练样本、`2^14`
个验证样本、embedding `Normal(0, 0.02)`、lm_head 全零初始化、无 bias、
constant LR、无 warmup/weight decay/gradient clipping，以及训练 200 iter。

历史 Transformer、FineWeb、freeze 和超参数 sweep 配置保存在
`legacy_presets.py` 的 `LEGACY_EXPERIMENTS` 中。它们只用于查阅和恢复；
若要重新运行，应先选择性移回 `presets.py`，而不是由当前入口隐式加载。

## 本地运行

当前 full-batch 训练入口接受一个 **bare token = 预设名**，以及任意个
`--group.key=value` 临时覆盖单个字段：

```bash
cd toy_models
python train_fullbatch.py fullbatch-mse0-shuffled-2p17-gd
python train_fullbatch.py fullbatch-mse0-shuffled-2p17-adam
python train_fullbatch.py fullbatch-mse0-shuffled-2p17-gd \
  --lr.learning_rate=3e-4 \
  --train.run_name=fullbatch-mse0-shuffled-2p17-gd-lr3e-4
```

## SCO 提交实验

当前入口是 `submit_sco_fullbatch.py`。它在单个 H100 节点上为 GD 和 Adam
各分配一张 GPU，并分别调用 `train_fullbatch.py`。

```bash
python3 toy_models/submit_sco_fullbatch.py
python3 toy_models/submit_sco_fullbatch.py --yes
```

## 新增 / 修改预设

在 `config/presets.py` 的 `EXPERIMENTS` 中复制当前最接近的配置，并确保
字典 key、`name`、`train.run_name` 与 `analyze.files_name` 相互对应。历史配置
若重新启用，也应选择性复制回来，不要整体重新注册。

## 各字段含义(改哪调哪,见 `config/schema.py`)

* **模型结构** `ModelConfig`:`n_layer`(0=仅 embed+head,1=单层,5=五层)、
  `block_type`(`transformer`=attn+FFN,`mlp`=FFN+FFN 无 attention)、`n_embd`、`n_head`、
  `head_dim`、`n_ffn`、`vocab_size`、`block_size`、`linear_bias`（统一控制所有
  attention / FFN / lm_head 的 `nn.Linear` 是否带 bias，默认 false；embedding
  本身无 bias）、`norm_eps`（所有 block 中 RMSNorm 的数值稳定项）
* **独立初始化** `ModelConfig`：`tok_emb_init_mean` / `tok_emb_init_std` 与
  `lm_head_init_mean` / `lm_head_init_std` 分别控制 embedding 和 lm_head 的
  Normal 初始化；`std=0` 时对应矩阵恒等于所设 mean。
* **数据/batch** `DataConfig`：`dataset`、`batch_size`（每卡；有效 batch = ×world_size）、
  `format`（`dual_stream` = synth 双流 `train_x.bin`/`train_y.bin`，默认；`nanogpt_shards` =
  FineWeb 单流 shard，header 后连续 token，target 右移一位）
* **优化器 + 超参** `OptimConfig`:`name`(sgd/adamw/adam)、`momentum`、`nesterov`(SGD)、
  `betas`、`eps`(Adam(W))、`weight_decay`、`grad_clip`(0 关闭裁剪)
* **学习率/调度** `LRConfig`:`scheduler`(cosine/constant)、`learning_rate`(峰值)、
  `min_lr`、`warmup_iters`(0 关闭 warmup)
* **训练循环 / checkpoint** `TrainConfig`:`max_iters`、`ckpt_fracs`、`run_name`、
  `eval_interval`、`eval_iters`、`log_interval`、`seed`
* **Hessian 分析** `AnalyzeConfig`:`max_classes`(lm_head 前 N 个 token 块)、
  `max_tokens`(embedding 前 N 个 token 块)、`n_batches`、`batch_size`、`num_bins`、
  `files_name`、`seed`

### 注意事项

* `ckpt_fracs` 是 dict,**不能**用 CLI / `EXP_ARGS` 覆盖(会报错),要改只能在预设里改。
  其它标量字段都能用 `--group.key=value` 临时覆盖。
* 覆盖时用**点号全路径**:`--optim.name=adamw`、`--lr.learning_rate=3e-4`、
  `--train.max_iters=8000`、`--analyze.max_classes=1024`,不能只写 `--max_iters=...`。
* `betas` 这类 tuple 字段用逗号写:`--optim.betas=0.9,0.999`。
