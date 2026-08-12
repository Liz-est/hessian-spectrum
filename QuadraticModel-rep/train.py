"""
B=64 训练脚本（8×H100 DDP），复现 QuadraticModel olmo150m 论文设置。

对齐 sweep.sh 的 cosine B=64 最优 run (qo82arcx, base_lr=2.0 → eta=opt.lr=16.0)：
  tokens=3B, seq_len=1024, batch=64 → tok/step=65536, steps=45776
  opt.lr=16.0, b1=0.9, b2_pct=0.01, eps=1e-8
  cosine schedule: warmup_pct=0.1, init=0.1, peak=1.0, end=0.1
  ema.pct=(0.04, 0.08)
  参数 fp32 主副本，forward bf16 autocast（与 JAX params-fp32/compute-bf16 一致）

checkpoint 在 10%/50%/100% 保存：模型参数 + 优化器 (nu/mu/log_prod_b2/pre/post/base_lr)
 + EMA，供后续 Lanczos 谱与 Adam 预条件器精确重建。

数据：data/fineweb_edu_bpe8192/{train,val}.bin（uint16，bpe_8192 重分词的 FineWeb-Edu）。

启动：
  torchrun --standalone --nproc_per_node=8 train.py
"""
import os, sys, time, math, pickle
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention import sdpa_kernel, SDPBackend

sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer, TransformerConfig
from opt import CompletePAdam, EMA

# ---------------- 配置 ----------------
DATA_DIR = "/data/250010020/hessian-spectrum/data/fineweb_edu_bpe8192_clean"
OUT_DIR = "/data/250010020/hessian-spectrum/QuadraticModel-rep/checkpoints_b64"
TOTAL_TOKENS = 3_000_000_000
BATCH = 64                 # 全局 batch（跨所有 GPU）
SEQ = 1024
OPT_LR = 16.0              # eta = sqrt(64)*2.0
B1, B2_PCT, EPS = 0.9, 0.01, 1e-8
WARMUP_PCT, INIT_V, PEAK_V, END_V = 0.1, 0.1, 1.0, 0.1
EMA_PCT = (0.04, 0.08)
CKPT_FRACS = (0.10, 0.50, 1.00)
SEED = 0
EVAL_EVERY_FRAC = 1 / 30   # 约 30 个 eval 点
EVAL_BATCHES = 40


def is_master():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def log(*a):
    if is_master():
        print(*a, flush=True)


class TokenLoader:
    """从连续 uint16 token 流按随机偏移采样 (B, SEQ+1) 窗口。"""
    def __init__(self, path, seq, device, seed):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq = seq
        self.device = device
        self.g = torch.Generator().manual_seed(seed)
        self.n = len(self.data)

    def batch(self, bs):
        ix = torch.randint(self.n - self.seq - 1, (bs,), generator=self.g)
        x = torch.stack([torch.from_numpy(self.data[i:i+self.seq].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i+1:i+1+self.seq].astype(np.int64)) for i in ix])
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def main():
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        rank, world, local = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert BATCH % world == 0, f"batch {BATCH} 不能整除 world {world}"
    local_bs = BATCH // world
    tok_per_step = BATCH * SEQ
    steps = (TOTAL_TOKENS + tok_per_step - 1) // tok_per_step
    ckpt_steps = {int(round(f * steps)): f for f in CKPT_FRACS}
    eval_steps = set(int(round(k * EVAL_EVERY_FRAC * steps)) for k in range(31))
    log(f"world={world} local_bs={local_bs} tok/step={tok_per_step} steps={steps}")
    log(f"checkpoint steps: {sorted(ckpt_steps)}")

    torch.manual_seed(SEED + rank)
    cfg = TransformerConfig(V=8192, seq_len=SEQ)
    model = Transformer(cfg).to(device).float()
    raw = model
    if ddp:
        model = DDP(model, device_ids=[local])

    # 优化器与 EMA 建在底层参数上
    lr_groups = raw.lr_groups()
    opt = CompletePAdam(lr_groups, lr=OPT_LR, total_steps=steps,
                        b1=B1, b2_pct=B2_PCT, eps=EPS,
                        schedule_kwargs=dict(warmup_pct=WARMUP_PCT, init_value=INIT_V,
                                             peak_value=PEAK_V, end_value=END_V))
    ema = EMA(list(raw.named_parameters()), pct=EMA_PCT) if is_master() else None
    # ν 的 EMA（与参考 nu_ema_state 对齐）：谱分析用 EMA'd ν 重建 Adam 预条件器
    nu_named = [(name, opt.nu[i]) for i, (name, _, _, _) in enumerate(lr_groups)]
    nu_ema = EMA(nu_named, pct=EMA_PCT) if is_master() else None

    train = TokenLoader(os.path.join(DATA_DIR, "train.bin"), SEQ, device, SEED + rank)
    val = TokenLoader(os.path.join(DATA_DIR, "val.bin"), SEQ, device, 12345 + rank)

    if is_master():
        os.makedirs(OUT_DIR, exist_ok=True)

    def save_ckpt(step, frac):
        if not is_master():
            return
        path = os.path.join(OUT_DIR, f"ckpt_p{int(frac*100)}.pt")
        torch.save(dict(
            step=step, frac=frac,
            model={k: v.detach().cpu() for k, v in raw.state_dict().items()},
            opt=opt.state_for_ckpt(),
            ema=ema.state_for_ckpt() if ema else None,
            nu_ema=nu_ema.state_for_ckpt() if nu_ema else None,
            log_prod_b2=opt.log_prod_b2,   # 供 EMA'd ν 的 bias correction
            config=dict(D=cfg.D, L=cfg.L, M=cfg.M, H=cfg.H, K=cfg.K, V=cfg.V,
                        seq_len=cfg.seq_len, batch=BATCH, opt_lr=OPT_LR,
                        total_tokens=TOTAL_TOKENS, steps=steps),
        ), path)
        log(f"  [saved] step {step} ({frac:.0%}) -> {path}")

    @torch.no_grad()
    def eval_loss():
        """返回 (val_loss, mean_entropy)。entropy 是 lm_head 曲率的核心止损指标：
        softmax 越尖锐(entropy 越低)→ lm_head GN 块 λ_max 越大(见口径对拍结论)。"""
        raw.eval()
        tot = torch.zeros((), device=device)
        ent = torch.zeros((), device=device)
        for _ in range(EVAL_BATCHES):
            x, y = val.batch(local_bs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, l = raw(x, y)
            tot += l.detach()
            p = F.softmax(logits.float(), dim=-1)
            ent += -(p * p.clamp_min(1e-12).log()).sum(-1).mean().detach()
        raw.train()
        if ddp:
            dist.all_reduce(tot, op=dist.ReduceOp.SUM); tot /= world
            dist.all_reduce(ent, op=dist.ReduceOp.SUM); ent /= world
        return (tot / EVAL_BATCHES).item(), (ent / EVAL_BATCHES).item()

    log("开始训练...")
    raw.train()
    t0 = time.time()
    loss_log = []   # (step, frac, val_loss, entropy, head_std) 供止损分析
    def head_std():
        with torch.no_grad():
            return float(raw.head.detach().float().std())
    # 存 init ckpt（step 0，未训练）——旧 run 缺失，谱分析需要基线
    if is_master():
        save_ckpt(0, 0.0)
    for step in range(steps + 1):
        if step in eval_steps or step in ckpt_steps:
            val, ent = eval_loss()
            hs = head_std()
            log(f"step {step:6d}/{steps} ({step/steps:.1%})  val_loss={val:.4f}  "
                f"entropy={ent:.3f}  head_std={hs:.4f}  elapsed={time.time()-t0:.0f}s")
            if is_master():
                loss_log.append((step, step/steps, val, ent, hs))
        if step in ckpt_steps:
            save_ckpt(step, ckpt_steps[step])
        if step == steps:
            break

        x, y = train.batch(local_bs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()   # DDP 在此 all-reduce（mean）梯度
        opt.step()
        if is_master():
            ema.update(raw.named_parameters())
            nu_ema.update([(name, opt.nu[i]) for i, (name, _, _, _) in enumerate(lr_groups)])

    log(f"训练完成，总耗时 {time.time()-t0:.0f}s")
    if is_master():
        import csv
        with open(os.path.join(OUT_DIR, "loss_log.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "frac", "val_loss", "entropy", "head_std"])
            w.writerows(loss_log)
        log(f"  [saved] loss_log.csv ({len(loss_log)} 行)")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
