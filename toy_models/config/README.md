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
* `config/presets.py` — 命名预设字典 `EXPERIMENTS`,和默认名 `DEFAULT`。
* `config/__init__.py` — `load(name)` 返回预设的深拷贝;`apply_overrides(cfg, argv)`
  应用 CLI 覆盖。

## 现有预设(`config/presets.py`)

| 预设名 | 模型 | 优化器 | run_name / files_name |
|---|---|---|---|
| `imbalance_s1_sgd`(默认) | 单层 vanilla,n_layer=1 | SGD | `vanilla_imbalance_s1-sgd` |
| `imbalance_s1_adamw` | 单层 vanilla,n_layer=1 | AdamW | `vanilla_imbalance_s1-adamw` |
| `simpliest_sgd` | 仅 embed+lm_head,n_layer=0 | SGD | `simpliest_imbalance_s1-sgd` |

`vanilla` 脚本对默认 `imbalance_s1_sgd`,`simpliest` 脚本对默认 `simpliest_sgd`。

## 本地运行

train / analyze 脚本接受两类 CLI:一个 **bare token = 预设名**(整体替换默认预设),
以及任意个 `--group.key=value` 覆盖单个字段。

```bash
cd toy_models
python train_vanilla_transformer.py                          # 默认预设 imbalance_s1_sgd
python train_vanilla_transformer.py imbalance_s1_adamw       # 换成另一个预设
python train_vanilla_transformer.py --optim.name=adamw --lr.learning_rate=3e-4
python train_simpliest_model.py                              # 默认预设 simpliest_sgd
python analyze_simpliest.py --analyze.max_classes=1024       # 全词表 lm_head/embedding
torchrun --standalone --nproc_per_node=8 train_vanilla_transformer.py   # 8 卡 DDP
```

## SCO 提交实验

两个提交脚本各自绑定一套模型和它的 train/analyze 脚本:

* `submit_sco_vanilla.py`   → `train_vanilla_transformer.py` + `analyze_vanilla.py`(默认 `imbalance_s1_sgd`)
* `submit_sco_simpliest.py` → `train_simpliest_model.py` + `analyze_simpliest.py`(默认 `simpliest_sgd`)

```bash
cd /data/250010020/hessian-spectrum
python3 toy_models/submit_sco_simpliest.py          # 问 y/n 再提交
python3 toy_models/submit_sco_simpliest.py --yes    # 直接提交
```

提交前记得把脚本里的 `JOB_NAME` 改成唯一名字。

### 提交时换/覆盖预设

改提交脚本里的 `EXP_ARGS`(train 和 analyze 两阶段共用同一份,保证 `run_name` /
checkpoint 计划一致):

```python
EXP_ARGS = ""                        # 留空 = 用脚本各自的默认预设
EXP_ARGS = "simpliest_sgd"           # 用某个命名预设(bare token)
EXP_ARGS = "--optim.name=adamw"      # 临时覆盖单个字段
EXP_ARGS = "imbalance_s1_adamw --lr.learning_rate=3e-4"   # 两者叠加
```

注意:bare token 会**整体替换**默认预设。在 `submit_sco_simpliest.py` 里若写成
一个 n_layer=1 的预设名(如 `imbalance_s1_sgd`),simpliest 脚本就会去建带
transformer block 的模型——别写错名字。

## 新增 / 修改预设

改 `config/presets.py` 的 `EXPERIMENTS` 字典:复制一个已有 block,起唯一 key,只改要变
的字段,并让 `run_name` / `files_name` 和 key 对应(避免和别的实验撞输出目录)。例如加个
AdamW 版的 simpliest:

```python
"simpliest_adamw": ExperimentConfig(
    name="simpliest_adamw",
    model=copy.deepcopy(_MODEL_EMBED_HEAD),      # n_layer=0
    data=DataConfig(dataset="synth_zipf_imbalanced_s1_V1024", batch_size=64),
    optim=OptimConfig(name="adamw", betas=(0.9, 0.95), weight_decay=0.1, grad_clip=1.0),
    lr=LRConfig(scheduler="cosine", learning_rate=6e-4, min_lr=3e-5, warmup_iters=200),
    train=TrainConfig(max_iters=8000, run_name="simpliest_imbalance_s1-adamw",
                      ckpt_fracs=dict(_CKPT_9)),
    analyze=AnalyzeConfig(files_name="simpliest_imbalance_s1-adamw"),
),
```

然后本地 `python train_simpliest_model.py simpliest_adamw`,或提交时 `EXP_ARGS = "simpliest_adamw"`。

## 各字段含义(改哪调哪,见 `config/schema.py`)

* **模型结构** `ModelConfig`:`n_layer`(0=仅 embed+head,1=单层)、`n_embd`、`n_head`、
  `head_dim`、`n_ffn`、`vocab_size`、`block_size`
* **数据/batch** `DataConfig`:`dataset`、`batch_size`(每卡;有效 batch = ×world_size)
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

