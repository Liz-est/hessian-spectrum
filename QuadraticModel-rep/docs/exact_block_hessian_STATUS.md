# exact 块 Hessian：2026-08-18 收尾 & 明天的计划

## 目标

画出 **某一层 loss Hessian 的精确矩阵样子**（热图），而不是 Lanczos 的谱密度分布。
对标 `tmp/Hessian-structure`（论文 "Towards Quantifying the Hessian Structure of
Neural Networks"）的 exact Hessian 热图，但要能跑在 rep 的 167M 模型上。

当前目标块：`layer01.mlp_up`，按 GN trace 选 top-8 神经元 → 8×1024 = **8192 维**，
H 是 8192² fp64 ≈ 0.50 GiB。

## 已完成

- `src/analysis/exact_block_hessian.py` — 新增。核心是**零值叶子注入**：
  `_forward_injected`（:57）复刻 `model.forward`，但在第 l 层目标参数的选定输出神经元列上
  加一个 `(n_sel, d_in)` 的零值叶子 `delta`，然后对 `delta` 求二阶导。
  因为 W 只经 `out = a @ W` 进入网络、输入激活 a 不依赖 W，所以
  `∂²loss/∂delta²` 与 `∂²loss/∂W` 在这些坐标上**逐元素恒等**（不是近似）。
  好处：cotangent 只有 `(n_sel, d_in)` 那么小，而不是整个 stacked 叶子（mlp_up 是
  50M）。这是相对 `SharedBatchOp` 的全部加速来源。
- `InjectedBatchOp`（:95）接口对齐既有的 `SharedBatchOp`（prepare/apply_batched/release）。
  `is_grads_batched=True` 一次算 chunk 行。
- **正确性已验证**：`python src/analysis/exact_block_hessian.py --verify` 在 tiny 模型
  (D=8,L=2,M=8,...) 上对拍暴力全 Hessian，hessian 与 gn 两支
  `max|Δ|=0.000e+00`，batched vs 逐行也是 0。注入路径可信。
- H 存成 npz 再单独出图（`scripts/plot_exact_hessian.py`），画图样式可反复改而不用重算。
- `sco/submit_exact_hessian.sh` + `sco/run_exact_hessian_inner.sh` — SCO 提交，8 卡 torchrun。

## 正在跑（明天先看这个）

`pt-wu6sfjxk` / `exact-hessian-ckpt-p50-hessian-0819-0007`，8×H100，RUNNING。
产物 → `outputs/exact_hessian/ckpt_p50_layer01_mlp_up_hessian_<stamp>.npz`
日志 → `test_outputs/exact_hessian_ckpt_p50_hessian_0819_0007.log`

**它在以 chunk=2 慢慢磨，会很久，但结果是对的**（数值正确性与 chunk 无关）。
明天要么它已经出了 npz 可以直接画图，要么按下面改完重跑更快。

## 两个待修问题（诊断已完成，未动手）

日志现象：GPU 利用率恒 12.5%、显存却远没吃满、但 chunk 从 256 一路减半到 2 都报 OOM。
这是**两件独立的事**：

### 1. 利用率 12.5% —— 8 张卡在算同一批行（最浪费的一处）

`make_local_batches`（`src/spectrum/spectrum_ddp.py:183`）确实按 rank 切了 minibatch，
所以数据是分开的、all_reduce 归一化也是对的。但
`dense_block_matrix_injected` 的行循环 `while i < n`（:172）是**全量**的 —— 每张卡都
把 8192 行整个算了一遍。当前配置 `n_tokens=65536, per=8` → `nb_global=8`、每卡
`local batches=1`，于是每张卡各自付一次昂贵的 `prepare()`，只为摊 1/8 的 token，
却重复了 100% 的行计算。

修法：行按 rank 切，`for i in range(rank*chunk, n, world*chunk)`，再 all_reduce 拼。
注意与现有的 minibatch 切分**不要重复切**：要么按行切（每卡看全部 minibatch），
要么按 minibatch 切（现状），混着切会把归一化分母搞错。倾向改成
**按行切 + 每卡遍历全部 minibatch**，因为行数（8192）比 minibatch 数（8）好分。

### 2. chunk 减到 2 还 OOM —— 是碎片，不是容量不够

显存分两块，只有一块随 chunk 缩：

| 来源 | per=8 时大小 | 随 chunk 缩？ |
|---|---|---|
| `prepare()` 的二阶图（MATH sdpa 把 `(B,H,T,T)` 注意力显式物化，12 层 × scores+probs × create_graph 再一份） | ~30 GiB | **否** |
| `is_grads_batched` 每条 lane 复制的 logits 尺寸中间量 | chunk × 0.25 GiB | 是 |

即真实占用 ≈ `30 + 0.25×chunk` GiB。80 GiB 卡上 chunk=8 就该塞得下，**不该在 chunk=2 还 OOM**。

根因：`except torch.OutOfMemoryError` 里只调了 `empty_cache()`（:181），但 `op` 还活着，
那 30 GiB 是活跃引用、`empty_cache()` 收不走。第一次 chunk=256 试图额外要 64 GiB
把分配器地址空间搅碎，之后每次减半都在**同一个已碎的分配器 + 同一张 op 图**上重试。
降到 2 能过不是因为终于够了，是因为要的块小到能塞进碎片缝。日志里
`free 7.40 / allocated 68.13 / reserved-but-unallocated 2.91 GiB` 就是这个形状。

修法（两条都做）：
- OOM 回退时先 `op.release()` + `empty_cache()`，再用新 chunk 重新 `prepare()`，
  别在脏分配器上原地重试。
- 更根本的是别依赖回退：**降 per 到 1~2、chunk 开大**。per=8 是把 30 GiB 固定开销
  放大了 8 倍去换一个用不上的并行度；per=1 时固定开销降到 ~3.75 GiB，chunk=256 时
  vmap 那块 8 GiB，合计 ~12 GiB，又快又不碰边。**token 总数靠多给 minibatch 补**
  （串行加和，不占额外显存），不靠加大 per。
- 可选：设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片。

## 明天的执行顺序

1. 看 `pt-wu6sfjxk` 状态和日志。若已出 npz → 直接 `scripts/plot_exact_hessian.py` 看图，
   确认热图形状符合预期（对角带 / block 结构）。
2. 改 `dense_block_matrix_injected`：行按 rank 切 + OOM 回退重建 op。
3. 改 inner 脚本参数：`--per 1 --chunk 256`，`--n_tokens` 保持 65536（靠 minibatch 数补）。
4. `--verify` 重跑一遍确认没改坏（tiny 模型、CPU 上跑，很快）。
5. 停掉旧 job，重提。预期从"chunk=2 单卡 8192 行"变成 8 卡各 1024 行、chunk=256。
6. 图 OK 后把 4 个新文件 + 2 个改动文件提交（当前分支 `quadratic-rep-senmiao`，均未提交）。

## 约束提醒

- **不要在 CPU 上跑大实验**（CPU 资源有限）。只有 `--verify` 这种 tiny 模型正确性
  检查可以本地跑。
- MATH sdpa 是硬要求：flash/efficient/cudnn 没有二阶反向。见 `src/spectrum/hvp.py` 顶部注释。
