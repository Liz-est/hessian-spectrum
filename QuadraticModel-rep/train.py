"""
B=64 训练脚本（8×H100 DDP），复现 QuadraticModel olmo150m 论文设置。

超参与模型 config 统一由 config.py 的 preset 管理（可切换优化器 Adam/Muon）：
  torchrun --standalone --nproc_per_node=8 train.py <preset名>
默认 preset = "b64_adam"（= 原设置）。"b64_muon" 走 CompletePMuon。

对齐 sweep.sh 的 cosine B=64 最优 run (qo82arcx, base_lr=2.0 → eta=opt.lr=16.0)：
  tokens=3B, seq_len=1024, batch=64 → tok/step=65536, steps=45776
  opt.lr=16.0, cosine schedule: warmup_pct=0.1, init=0.1, peak=1.0, end=0.1
  ema.pct=(0.04, 0.08)；参数 fp32 主副本，forward bf16 autocast

checkpoint 在 10%/50%/100% 保存：模型参数 + 优化器状态（Adam: nu/mu/log_prod_b2；
Muon: 动量 buffer）+ 参数 EMA + 优化器状态 EMA（Adam 存 nu_ema、Muon 存 buf_ema），
供后续 Lanczos 谱与预条件器精确重建。config dict 记 optim_name 供谱分析自动切换。

数据：grain 采样（复现 QuadraticModel/data.py 管线，源换 parquet）。读 100BT parquet
（train=files[:-1] / eval=files[-1:]），滑窗 shuffle → 逐文档分词追加 <eot>
→ ConcatThenSplit 切 (seq_len+1) 窗 → 顺序 batch。见 data_grain.build_loaders。
⚠ parquet 源用滑窗 shuffle（非原版全局索引 shuffle），文档顺序不同、非 bit 级复现。
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
from opt import CompletePAdam, CompletePMuon, EMA
from config import load, build_optimizer
from data_grain import build_loaders


def is_master():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def log(*a):
    if is_master():
        print(*a, flush=True)


def optimizer_state_named(opt, lr_groups):
    """返回 [(name, tensor)]：Adam 用 ν、Muon 用动量 buffer。供 EMA 与谱分析。"""
    if isinstance(opt, CompletePAdam):
        return [(name, opt.nu[i]) for i, (name, _, _, _) in enumerate(lr_groups)]
    return [(name, opt.buf[i]) for i, (name, _, _, _) in enumerate(lr_groups)]


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "b64_adam"
    cfg = load(preset)
    log(f"preset={preset}  optim={cfg.optim.name}  batch={cfg.batch_size}  out={cfg.out_dir}")

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

    BATCH, SEQ = cfg.batch_size, cfg.seq_len
    assert BATCH % world == 0, f"batch {BATCH} 不能整除 world {world}"
    local_bs = BATCH // world
    tok_per_step = BATCH * SEQ
    steps = (cfg.total_tokens + tok_per_step - 1) // tok_per_step   # schedule 的 full steps（不改）
    ckpt_steps = {int(round(f * steps)): f for f in cfg.ckpt_fracs}
    # eval 点：优先 eval_every_steps（lr 扫描用密集网格），否则 eval_every_frac 的 31 点。
    if cfg.eval_every_steps > 0:
        eval_steps = set(range(0, steps + 1, cfg.eval_every_steps))
    else:
        eval_steps = set(int(round(k * cfg.eval_every_frac * steps)) for k in range(31))
    # 实际循环步数：max_train_steps>0 则提前停（schedule 不变，只是早停做早段对比）。
    run_steps = cfg.max_train_steps if cfg.max_train_steps > 0 else steps
    eval_steps.add(run_steps)
    log(f"world={world} local_bs={local_bs} tok/step={tok_per_step} "
        f"schedule_steps={steps} run_steps={run_steps}")
    log(f"checkpoint steps: {sorted(ckpt_steps)}")

    torch.manual_seed(cfg.seed + rank)
    model_cfg = cfg.model
    model = Transformer(model_cfg).to(device).float()
    raw = model
    if ddp:
        model = DDP(model, device_ids=[local])

    # 优化器与 EMA 建在底层参数上（按 config 切换 Adam/Muon）
    lr_groups = raw.lr_groups()
    opt = build_optimizer(raw, cfg, steps)
    optim_name = cfg.optim.name.lower()
    ema = EMA(list(raw.named_parameters()), pct=cfg.ema_pct) if is_master() else None
    # 优化器状态 EMA（Adam: ν 的 EMA；Muon: 动量 buffer 的 EMA）。谱分析用它重建预条件器。
    ostate_named = optimizer_state_named(opt, lr_groups)
    ostate_ema = EMA(ostate_named, pct=cfg.ema_pct) if is_master() else None

    # 数据：原版 grain 采样（seed 全局 shuffle + concat-split + 顺序 batch）。
    train_iter, eval_factory, _ds = build_loaders(
        data_dir=cfg.data_dir,
        seq_len=SEQ, vocab_size=cfg.vocab,
        global_batch=BATCH, eval_batch=BATCH,
        device=device, rank=rank, world=world, seed=cfg.seed,
    )

    if is_master():
        os.makedirs(cfg.out_dir, exist_ok=True)

    def save_ckpt(step, frac):
        if not is_master():
            return
        path = os.path.join(cfg.out_dir, f"ckpt_p{int(frac*100)}.pt")
        # 优化器状态 EMA：Adam 存 nu_ema，Muon 存 buf_ema（键名区分，谱分析按 optim_name 取）
        ostate_ema_key = "nu_ema" if optim_name == "adam" else "buf_ema"
        ck = dict(
            step=step, frac=frac,
            model={k: v.detach().cpu() for k, v in raw.state_dict().items()},
            opt=opt.state_for_ckpt(),
            ema=ema.state_for_ckpt() if ema else None,
            log_prod_b2=getattr(opt, "log_prod_b2", 0.0),   # Adam 用；Muon 无（0.0）
            config=dict(D=model_cfg.D, L=model_cfg.L, M=model_cfg.M, H=model_cfg.H,
                        K=model_cfg.K, V=model_cfg.V, seq_len=model_cfg.seq_len,
                        batch=BATCH, opt_lr=cfg.opt_lr, optim_name=optim_name,
                        total_tokens=cfg.total_tokens, steps=steps),
        )
        ck[ostate_ema_key] = ostate_ema.state_for_ckpt() if ostate_ema else None
        torch.save(ck, path)
        log(f"  [saved] step {step} ({frac:.0%}) -> {path}")

    @torch.no_grad()
    def eval_loss():
        """返回 (val_loss, mean_entropy)。entropy 是 lm_head 曲率的核心止损指标：
        softmax 越尖锐(entropy 越低)→ lm_head GN 块 λ_max 越大(见口径对拍结论)。
        eval_factory() 每次返回从头开始的有限 eval 流（shuffle=False），最多取
        EVAL_BATCHES 个 batch；不足则用实际取到的数量归一。"""
        raw.eval()
        tot = torch.zeros((), device=device)
        ent = torch.zeros((), device=device)
        eval_it = eval_factory()
        n = 0
        for _ in range(cfg.eval_batches):
            try:
                x, y = next(eval_it)
            except StopIteration:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, l = raw(x, y)
            tot += l.detach()
            p = F.softmax(logits.float(), dim=-1)
            ent += -(p * p.clamp_min(1e-12).log()).sum(-1).mean().detach()
            n += 1
        raw.train()
        n = max(n, 1)
        if ddp:
            dist.all_reduce(tot, op=dist.ReduceOp.SUM); tot /= world
            dist.all_reduce(ent, op=dist.ReduceOp.SUM); ent /= world
        return (tot / n).item(), (ent / n).item()

    log("开始训练...")
    raw.train()
    t0 = time.time()
    loss_log = []   # (step, frac, val_loss, entropy, head_std) 供止损分析
    def head_std():
        with torch.no_grad():
            return float(raw.head.detach().float().std())
    # 存 init ckpt（step 0，未训练）——旧 run 缺失，谱分析需要基线。扫描模式（无 ckpt_fracs）跳过。
    if is_master() and cfg.ckpt_fracs:
        save_ckpt(0, 0.0)
    for step in range(run_steps + 1):
        if step in eval_steps or step in ckpt_steps:
            vloss, ent = eval_loss()
            hs = head_std()
            log(f"step {step:6d}/{steps} ({step/steps:.1%})  val_loss={vloss:.4f}  "
                f"entropy={ent:.3f}  head_std={hs:.4f}  elapsed={time.time()-t0:.0f}s")
            if is_master():
                loss_log.append((step, step/steps, vloss, ent, hs))
        if step in ckpt_steps:
            save_ckpt(step, ckpt_steps[step])
        if step == run_steps:
            break

        x, y = next(train_iter)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()   # DDP 在此 all-reduce（mean）梯度
        opt.step()
        if is_master():
            ema.update(raw.named_parameters())
            ostate_ema.update(optimizer_state_named(opt, lr_groups))

    log(f"训练完成，总耗时 {time.time()-t0:.0f}s")
    if is_master():
        import csv
        with open(os.path.join(cfg.out_dir, "loss_log.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "frac", "val_loss", "entropy", "head_std"])
            w.writerows(loss_log)
        log(f"  [saved] loss_log.csv ({len(loss_log)} 行)")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
