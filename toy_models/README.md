# Toy Models

用小型 decoder-only Transformer 研究 **优化器 (AdamW vs SGD) × 数据分布 (balance /
imbalance / fineweb) × 模型深度** 对训练动态和 Hessian 谱的影响
（论文 arXiv 2402.16788 的 toy 化扩展）。

两条 Hessian 分析路线并存：

1. **逐单元精确块谱**（`analyze_vanilla.py` + `hessian_toy.py`）——按
   头 / 神经元 / token 切块，精确特征分解，看块内谱与异质性（block-diagonal 视角）；
2. **全参数 SLQ 谱**（`analyze_full_spectrum.py`）——对**整个参数向量**的
   Hessian 做随机 Lanczos 求积（含跨块曲率），得到真正的全 Hessian ESD。

---

## 文件结构

```
toy_models/
├── config/                      # 配置体系（所有入口脚本共用）
│   ├── schema.py                #   dataclass 定义：Model/Data/Optim/LR/Train/AnalyzeConfig
│   ├── presets.py               #   EXPERIMENTS dict：preset 名 -> 完整 ExperimentConfig
│   ├── build.py                 #   build_optimizer / make_lr_fn（config -> torch 对象）
│   └── __init__.py              #   load(preset) + apply_overrides(--group.key=value)
│
├── vanilla_model.py             # ToyVanilla：Post-LN + sinusoidal PE + ReLU FFN（方案 C）
│                                #   支持 block_type="mlp"（attention 槽换成第二个 FFN）
│                                #   与 loss_type="ce"/"mse"
├── simpliest_model.py           # ToyVanilla 变体：forward 不加位置编码；n_layer=0
│                                #   即 embed+lmhead-only。⚠ 与 vanilla_model state_dict
│                                #   相同但 forward 不同，load ckpt 必须按 run 名选类
├── llama_model.py / llama_A.py / llama_B.py   # LLaMA-style 方案 A/B（仅参数量参考）
├── vanilla_transformer.py       # 方案 C 的 build()，直接运行打印参数量分解
│
├── train_vanilla_transformer.py # 训练入口（torchrun DDP；写 ckpt + loss 曲线）
├── train_simpliest_model.py     # simpliest 系的训练入口
│
├── hessian_toy.py               # 【库】精确 per-unit Hessian 块 + 特征分解 + hetero 距离
├── analyze_vanilla.py           # 【入口】逐单元块谱分析（依赖 hessian_toy）
├── analyze_simpliest.py         # simpliest 系的对应分析入口
├── analyze_full_spectrum.py     # 【入口】全参数 Hessian ESD（SLQ，自包含，不依赖 hessian_toy）
│
├── eval_ckpts_val.py            # 对已有 run 的 ckpt 事后补算 val loss
├── compute_fineweb_token_freq.py# 生成 token_counts.npy（token_select="freq" 用）
│
├── submit_sco_vanilla.py        # SCO 单 job 提交：训练 -> 块谱分析 [-> SLQ]（见下）
├── submit_sco_simpliest.py      # simpliest 系对应提交脚本
├── sco_run_*.py                 # 各批次实验的参数写死的提交/排队编排器（历史存档）
│
├── runs/<run_name>/             # 训练产物：ckpt_<tag>.pt、loss_log.csv、loss 曲线
└── files/<files_name>/          # 分析产物：块谱 + hetero 图；slq/ 子目录放全参数谱
```

**入口 vs 库**：`train_*.py`、`analyze_*.py`、`eval_*.py`、`submit_*.py`、`sco_run_*.py`
是可执行入口；`hessian_toy.py`、`config/`、`*_model.py` 是被 import 的库，不直接运行。

## 配置体系与数据流

所有入口共用同一套 CLI：第一个裸参数选 preset（`config/presets.py` 里的
`EXPERIMENTS` key），`--group.key=value` 覆盖单个字段：

```bash
torchrun --standalone --nproc_per_node=8 train_vanilla_transformer.py layer5-imbalance-s1-adamw
python3 analyze_vanilla.py layer5-imbalance-s1-adamw --analyze.max_classes=1024
```

路径约定（两条分析路线都锚定在这两个字段上）：

- `train.run_name` → checkpoint 读/写目录 `runs/<run_name>/`
- `analyze.files_name` → 分析输出目录 `files/<files_name>/`

⚠ preset key ≠ 目录名（如 preset `balance_adamw` → `files/vanilla_balance-adamw`）。
任何按目录操作的脚本都应先 `config.load(preset)` 解析真实名字。

```
config/presets.py (preset) ──► train_vanilla_transformer.py ──► runs/<run_name>/ckpt_<tag>.pt
                                                                      │   (9 个 tag: init/p10/p25/
                    ┌─────────────────────────────────────────────────┤    p40/p50/p60/p75/p85/p100)
                    ▼                                                 ▼
      analyze_vanilla.py + hessian_toy.py                analyze_full_spectrum.py
      逐单元精确块谱（(ckpt × 层) 分片到 8 卡）           全参数 SLQ（(ckpt × probe) 分片到 8 卡）
                    │                                                 │
                    ▼                                                 ▼
      files/<files_name>/<tag>/eigs_*.npy                files/<files_name>/slq/<tag>/ritz_v*.npz
      files/<files_name>/<tag>/spectrum_*.png            files/<files_name>/slq/spectrum_full_*.png
      files/<files_name>/evolution_*.png                 files/<files_name>/slq/summary_slq.json
```

两条路线都支持**断点续跑**：块谱以 `summary_<layer>.json` 是否存在为完成标志，
SLQ 以 `ritz_v<k>.npz` 为标志——重跑同一命令会自动跳过已完成项。

---

## 模型

### 方案 C：vanilla decoder-only Transformer（核心工作模型）

| 配置                        | 值                    |
| --------------------------- | --------------------- |
| Vocabulary size `V`         | 1024（fineweb: 50304）|
| Hidden size `d`             | 192                   |
| Attention heads `h`         | 6                     |
| Head dimension `d_head`     | 32                    |
| FFN size `d_ff`             | 1024                  |
| Context length              | 128（fineweb: 1024）  |
| Position encoding           | Fixed sinusoidal      |
| Normalization               | Post-LayerNorm        |
| FFN activation              | ReLU                  |
| Weight tying                | False（untied）       |
| Linear bias                 | True                  |
| Dropout                     | 0                     |

深度是 preset 的自由维度：`n_layer ∈ {0 (simpliest/mse0), 1 (vanilla), 5 (layer5,
mlp10), 20 (layer20)}`。单层参数量分解（`python3 vanilla_transformer.py` 可打印）：

```
N_embed+head  = 2·V·d + V                  = 394,240   (V=1024)
N_block       = 4d² + 2·d·d_ff + 9d + d_ff = 543,424   (每层)
```

如 layer5（V=1024）总参数 3,111,360 ≈ 3.11M。

变体（都由 preset 的 model 字段控制，无需改代码）：

- **simpliest**（`simpliest_model.py`）：n_layer=0 且 forward 不加位置编码，
  纯 embed→lmhead。⚠ 与 vanilla_model 的 state_dict 完全相同，加载 ckpt 时按
  run 名开头是否为 `simpliest` 选模型类。
- **mlp10**：`block_type="mlp"`，5 个 block 的 attention 槽替换为第二个 FFN，
  即 10 个 FFN 子层、无 attention。
- **mse0**：n_layer=0 + `loss_type="mse"`（one-hot 目标的 MSE，走 vanilla_model）。

### 方案 A / B：LLaMA-style decoder（仅参数量参考，未在实验矩阵中使用）

单层 RMSNorm + RoPE + SwiGLU、无 bias、untied，V=1024：

| 配置 | A (llama_A) | B (llama_B) |
| ---- | ----------: | ----------: |
| `d` / `h` / `d_head` / `d_ff` | 160 / 5 / 32 / 448 | 256 / 4 / 64 / 680 |
| 总参数量 | 0.646M | 1.309M |

```
N_embed+head  = 2·V·d
N_transformer = 4d² + 3·d·d_ff + 3d
```

---

## 数据

- **合成 bigram 数据**（`<repo-root>/data/synth_*_V1024[_1B]/`，由
  `data_construction/` 生成，dual_stream 格式 `train_x/train_y.bin` + `meta.pkl`）：
  一阶马尔可夫链，`pi`（token 频率：uniform=balance / zipf=imbalance）与
  predictability `a`（任务难度）解耦。最优 loss 由链的条件熵率决定。
- **fineweb10B**（`data/fineweb10B/`，nanogpt_shards 格式）：真实数据，GPT-2 BPE。

preset 的 `data.dataset` / `data.format` 指定数据集；提交作业前数据必须已生成。

## 训练（`train_vanilla_transformer.py`）

torchrun DDP（8 卡，每卡 batch=64，有效 batch=512）；rank 0 负责 eval / 日志 /
写 checkpoint。超参全部来自 preset，典型值：

| 项 | 10M 合成数据 | 1B 合成数据 | fineweb10B |
| --- | --- | --- | --- |
| optimizer | preset 决定（adamw: β=(0.9,0.95)；sgd: momentum=0.9）；wd=0.1 |||
| lr | 6e-4，cosine → 3e-5（fineweb → 6e-5），warmup 200 | warmup 2000 | warmup 400 |
| max_iters | 8000（mlp10: 20000） | 130000 | 20000 |
| checkpoint | 9 点：init/p10/p25/p40/p50/p60/p75/p85/p100（`train.ckpt_fracs`）|||

产物：`runs/<run_name>/ckpt_<tag>.pt`（内含完整 `experiment` config dict）、
`loss_log.csv`、`val_loss_log.csv`、loss 曲线图。

---

## Hessian 路线一：逐单元精确块谱（`analyze_vanilla.py`）

模型很小，块内**不用随机方法**：对每个「单元」精确构造 Hessian / Gauss–Newton
块并 `torch.linalg.eigvalsh` 特征分解。不同层用不同分块粒度
（`hessian_toy.default_layer_spec`；多 block 时层名带 `b<i>_` 前缀）：

| 层 | 分块粒度 | 单元数 | 块尺寸 |
| --- | --- | --- | --- |
| `embedding`（`tok_emb`） | 按 token | `max_tokens`（默认 256） | 192² |
| `b<i>_attn_wq` / `_wk` | 按注意头 | 6 | 6144² |
| `b<i>_attn_wv` / `_attn_proj` | 按输出神经元 | 192 | 192² |
| `b<i>_mlp_fc` | 按输出神经元 | 1024 | 192² |
| `b<i>_mlp_proj` | 按输出神经元 | 192 | 1024² |
| `lm_head` | 按 token（=类别） | `max_classes`（默认 256） | 192² |

（mlp10 的 block 首槽换成 `b<i>_ffn1_fc/proj`、`b<i>_ffn2_fc/proj`，均按神经元。）

块的精确构造（利用「输出对该单元权重线性」，二阶项严格为零）：

- **按输出神经元**：`H_i = (1/N) Σ_t s_{i,t} x_t x_tᵀ`，`s_{i,t} = (∂L/∂y_{i,t})²`
  （empirical Fisher / GN，PSD）。
- **按注意头**：`H_h = (1/N) Σ_t (x_t x_tᵀ) ⊗ (g_{h,t} g_{h,t}ᵀ)`。
- **lm_head，CE**：`H_k = (1/N) Σ_t p_{k,t}(1−p_{k,t}) x_t x_tᵀ`（精确）。
- **lm_head，MSE**（`loss_type="mse"` 时自动切换）：`H = (2/C)·mean_t x_t x_tᵀ`，
  对所有类别相同——没有 CE 的 `p(1−p)` 类不平衡曲率。
- **embedding**：`H_v = (1/N_v) Σ_{t: x_t=v} g_t g_tᵀ`（Fisher，只在 v 出现的位置累加）。

每块特征分解后：(1) 汇成该层 ESD 图；(2) 单元谱两两算 Symmetric KL / JS
distance → hetero 矩阵/热图。

torchrun 下 `(checkpoint × 层)` 工作项按 rank strided 分片（如 layer5 为
9×32=288 项）；结束后 rank 0 统一渲染。

**产物**（`files/<files_name>/`）：

```
<tag>/                                    tag ∈ 9 个 checkpoint
  spectrum_<layer>.png / _log.png         各层块谱 ESD
  hetero_<layer>_{skl,js}.png             各层 hetero 热图
  eigs_<layer>.npy / hetero_<layer>_*.npy / summary_<layer>.json
evolution_{skl,js}.png                    hetero 均值随训练演变（每层一条线）
evolution_layers_{skl,js}.png             跨层视角
all_summary.json
```

fineweb 词表大（50304），preset 里用 `max_classes=1024 max_tokens=1024
token_select="freq"` 只分析最高频的 1024 个 token（需先跑
`compute_fineweb_token_freq.py` 生成 `token_counts.npy`）。

## Hessian 路线二：全参数 SLQ 谱（`analyze_full_spectrum.py`）

对**整个参数向量**（layer5 为 3.11M 维）的 Hessian 做谱密度估计——含跨块曲率，
这是块谱（block-diagonal 视角）给不出的。精确对角化在 3.1M 维不可行，采用
SLQ（随机 Lanczos 求积），并利用模型小做了两个精度升级（相对
`language_models/hessian_spectrum.py` 的论文原版实现）：

- **fp64 端到端**：模型转 double，HVP 与 Lanczos 递推全程 float64（3.1M 参数
  fp64 仅 ~25 MB）；
- **完全重正交化**：全部 m 个 Lanczos 基向量留在显存（m=100 × 3.1M × 8B ≈
  2.5 GB），每步对整个基做两次重正交，消除朴素三项递推的 ghost eigenvalue。

方法要点：

- HVP 用 double backward（`autograd.grad(loss, params, create_graph=True)` 再对
  `⟨g,v⟩` 求导）；SDPA 需强制 `sdpa_kernel(SDPBackend.MATH)`（flash 核无二阶反传）。
- Hessian 算子 = loss 在**固定 seed 生成的 `slq_n_batches` 个 batch** 上的平均，
  所以每个 probe / rank / checkpoint 面对的是同一个确定性矩阵。
- 每个 probe 跑 m 步 Lanczos → 三对角阵 T 特征分解 → Ritz 值 + 权重（首分量平方）
  → 高斯核（方差 `slq_sigma2`）重构连续密度，probe 间取平均。
- 工作项 `(checkpoint × probe)`（默认 9×16=144 个独立 Lanczos）按 rank 分片，
  **零通信**；rank 0 收尾渲染。

超参在 `AnalyzeConfig`：`slq_m=100`、`slq_num_v=16`、`slq_n_batches=64`、
`slq_sigma2=1e-5`、`slq_dtype="fp64"`。

**产物**（`files/<files_name>/slq/`）：

```
<tag>/ritz_v<k>.npz                 每个 probe 的 Ritz 值+权重（resume 标志）
density_<tag>.npz                   重构的谱密度曲线
spectrum_full_<tag>.png             单 checkpoint 全谱（semilogy）
spectrum_full_evolution.png         9 个 checkpoint 叠加演变
summary_slq.json                    每 tag 的 λ_max / λ_min / trace/D
```

```bash
# 单独跑（本地或集群；同一套 preset/override CLI）
torchrun --standalone --nproc_per_node=8 analyze_full_spectrum.py <preset> \
    [--train.run_name=... --analyze.files_name=...]
```

---

## SCO 提交：训练 + 分析一体化（`submit_sco_vanilla.py`）

提交单个 8×H100 作业，最多三阶段串行、共用同一组 config tokens
（保证 run_name / files_name / ckpt 调度一致）：

1. **训练**：`torchrun --nproc_per_node=8 train_vanilla_transformer.py <EXP_ARGS>`
2. **块谱分析**：`torchrun ... analyze_vanilla.py <EXP_ARGS>`
3. **全参数 SLQ**（可选）：`torchrun ... analyze_full_spectrum.py <EXP_ARGS>`
   ——仅当解析后的 config 有 **`analyze.slq=True`** 时追加。开关来源二选一：
   preset 里写 `AnalyzeConfig(..., slq=True)`，或在 `EXP_ARGS` 里临时加
   `--analyze.slq=true`。

```bash
# 编辑脚本顶部 JOB_NAME / EXP_ARGS 后：
python3 submit_sco_vanilla.py          # 打印解析出的 runs/files 目录与阶段列表，需确认
python3 submit_sco_vanilla.py --yes    # 直接提交
```

单机也可手动逐阶段跑：

```bash
cd toy_models
torchrun --standalone --nproc_per_node=8 train_vanilla_transformer.py <preset>
python3 analyze_vanilla.py <preset>
python3 analyze_full_spectrum.py <preset>        # 若要全参数谱
```

其他常用工具：

```bash
python3 eval_ckpts_val.py              # 对旧 run 的 ckpt 补算 val loss
/root/.sco/bin/sco --profile zhanglixian-g acp jobs list \
    --workspace-name p10-intelligent-adaptation-and-optimization-for-domestic-ai   # 查 job
```
