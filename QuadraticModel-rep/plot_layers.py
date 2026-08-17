"""
逐块（逐层）Hessian/GN 谱作图。参考 plot_compare_running.py 的曲线约定，但**不与论文对比**
（论文只有全量谱，没有逐块谱，没有可比对象）。

沿用的约定（与 plot_compare_running.py 一致）：
  - combined_positive：locked 正端(y>0) 与连续体(g>0,mid>0) 合并 → 按 index 排序 → 去重
  - log-log，横轴 eigenvalue index (rank)，纵轴 eigenvalue
  - 误差带取自连续体 lo/hi

⚠ 不能 import plot_compare_running：那个模块的模块体是可执行作图代码（import 即画图并
读它自己的 npz）。故把 combined_positive 复制过来（十行，没必要为此重构既有脚本）。

每条曲线出三张图：
  1. <tag>_overlay.png     所有块叠在一张，按深度着色；左=绝对 index，右=index/n_b 归一
  2. <tag>_grid.png        每块一个小面板
  3. <tag>_summary.png     λmax / trace / 有效秩 / 负特征值占比 按块（层）排列

用法:
  python plot_blocks.py --npz outputs/p100_grain/blocks_hessian_raw.npz --outdir outputs/p100_grain/fig
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

CURVE_TITLE = {
    "hessian_raw": "Raw Hessian (CompleteP)",
    "hessian_adam": "Preconditioned Hessian",
    "gn_raw": "Raw Gauss-Newton (CompleteP)",
    "gn_adam": "Preconditioned Gauss-Newton",
}


def combined_positive(cur):
    """与 plot_compare_running.combined_positive 同逻辑（复制，见文件头说明）。"""
    L = int(np.atleast_1d(cur["L"])[0])
    xs, ys = [], []
    if L:
        xl, yl = cur["x"][:L], cur["y"][:L]
        q = np.isfinite(xl) & np.isfinite(yl) & (yl > 0)
        xs.append(xl[q]); ys.append(yl[q])
    g, mid = cur["g"], cur["mid"]
    q = np.isfinite(g) & np.isfinite(mid) & (g > 0) & (mid > 0)
    xs.append(mid[q]); ys.append(g[q])
    x = np.concatenate(xs); y = np.concatenate(ys)
    order = np.argsort(x); x, y = x[order], y[order]
    uniq = np.r_[True, np.diff(x) > 0]
    return x[uniq], y[uniq]


def load_curve(npz, tag, block):
    pre = f"{tag}_{block}"
    keys = ["x", "y", "g", "mid", "lo", "hi", "L", "R"]
    if f"{pre}_x" not in npz.files:
        return None
    cur = {k: npz[f"{pre}_{k}"] for k in keys}
    cur["numel"] = int(npz[f"{pre}_numel"])
    cur["eigs"] = npz[f"{pre}_eigs"]
    return cur


def block_order_key(name):
    """embd 最前、head 最后，中间按层号排；同层内按张量的前向顺序。"""
    if name == "embd":
        return (-1, 0, name)
    if name == "head":
        return (10**6, 0, name)
    if name.startswith("layer"):
        body = name[5:]
        num = int(body.split(".")[0]) if body[:2].isdigit() else 0
        part = body.split(".", 1)[1] if "." in body else ""
        order = {"attn_q": 1, "attn_k": 2, "attn_v": 3, "attn_head": 4,
                 "mlp_up": 5, "mlp_head": 6, "attn": 1, "mlp": 5}
        return (num, order.get(part, 0), name)
    return (10**5, 0, name)


def block_kind(name):
    """块名 → 张量类别（用于 74 块时按类别着色/分面）。"""
    if "." in name:
        return name.split(".", 1)[1]
    return name


KIND_COLORS = {
    "attn_q": "#0072B2", "attn_k": "#56B4E9", "attn_v": "#009E73",
    "attn_head": "#E69F00", "mlp_up": "#D55E00", "mlp_head": "#CC79A7",
    "attn": "#0072B2", "mlp": "#D55E00",
    "embd": "#000000", "head": "#666666",
}


def discover(npz):
    """返回 {tag: [block_name, ...]}。"""
    out = {}
    for f in npz.files:
        if not f.endswith("_numel"):
            continue
        stem = f[: -len("_numel")]
        for tag in CURVE_TITLE:
            if stem.startswith(tag + "_"):
                out.setdefault(tag, []).append(stem[len(tag) + 1:])
    for tag in out:
        out[tag] = sorted(set(out[tag]), key=block_order_key)
    return out


def summarize(ev):
    ev = np.asarray(ev, dtype=np.float64)
    pos = ev[ev > 0]
    return {
        "lam_max": float(ev.max()) if ev.size else np.nan,
        "lam_min": float(ev.min()) if ev.size else np.nan,
        "trace": float(ev.sum()),
        "eff_rank": float(pos.sum() ** 2 / (pos ** 2).sum()) if pos.size else 0.0,
        "frac_neg": float((ev < 0).mean()) if ev.size else 0.0,
    }


def fig_overlay(npz, tag, blocks, outpath):
    """所有块叠一张。块多时（layer-tensors 74 块）按**张量类别**着色、层内深浅渐变，
    否则 74 条同色系曲线无法区分。"""
    kinds = [block_kind(b) for b in blocks]
    uniq_kinds = sorted(set(kinds), key=lambda k: block_order_key(f"layer00.{k}"))
    many = len(blocks) > 16
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    handles = []
    # 每个类别内部按层号给深浅
    per_kind_idx = {}
    for b in blocks:
        k = block_kind(b)
        per_kind_idx.setdefault(k, []).append(b)

    for b in blocks:
        cur = load_curve(npz, tag, b)
        if cur is None:
            continue
        k = block_kind(b)
        if many:
            base = KIND_COLORS.get(k, "#888888")
            idx = per_kind_idx[k].index(b)
            n_in = max(len(per_kind_idx[k]) - 1, 1)
            # 同类内按层号从浅到深（alpha 渐变）
            alpha = 0.35 + 0.65 * (idx / n_in)
            col, lw = base, 0.9
        else:
            col = plt.get_cmap("viridis")(blocks.index(b) / max(len(blocks) - 1, 1))
            lw, alpha = 1.2, 1.0
        x, y = combined_positive(cur)
        axes[0].plot(x, y, color=col, lw=lw, alpha=alpha)
        axes[1].plot(x / cur["numel"], y, color=col, lw=lw, alpha=alpha)
        if not many:
            handles.append(Line2D([0], [0], color=col, lw=1.5, label=b))
    if many:
        handles = [Line2D([0], [0], color=KIND_COLORS.get(k, "#888888"), lw=1.8,
                          label=k) for k in uniq_kinds]
    axes[0].set_xlabel("eigenvalue index (rank)")
    axes[0].set_title(f"{CURVE_TITLE.get(tag, tag)} — per-block spectra (absolute index)",
                      fontsize=10, loc="left")
    axes[1].set_xlabel("index / n_b  (normalized within block)")
    axes[1].set_title("Normalized by block dimension (shapes comparable)",
                      fontsize=10, loc="left")
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylabel("eigenvalue")
        ax.grid(True, which="both", color="#d4d4d4", alpha=0.55, lw=0.45)
    axes[0].legend(handles=handles, frameon=False, fontsize=7,
                   ncol=2 if many else 2, loc="lower left",
                   title="tensor" if many else None, title_fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved:", outpath, flush=True)


def fig_by_kind(npz, tag, blocks, outpath):
    """按张量类别分面：每个面板一类（attn_q/attn_k/.../mlp_head），
    面板内每条线是一层，用 colormap 表示深度 —— 直接看「同一张量随深度怎么变」。"""
    per_kind = {}
    for b in blocks:
        per_kind.setdefault(block_kind(b), []).append(b)
    # 只保留有多层的类别（embd/head 各只有一块，单独画没意义）
    kinds = [k for k in sorted(per_kind, key=lambda k: block_order_key(f"layer00.{k}"))
             if len(per_kind[k]) > 1]
    if not kinds:
        return
    ncol = min(3, len(kinds))
    nrow = (len(kinds) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.7 * nrow),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(kinds):
        ax = axes[i // ncol][i % ncol]
        ax.axis("on")
        bl = sorted(per_kind[k], key=block_order_key)
        for j, b in enumerate(bl):
            cur = load_curve(npz, tag, b)
            if cur is None:
                continue
            x, y = combined_positive(cur)
            ax.plot(x, y, color=cmap(j / max(len(bl) - 1, 1)), lw=1.0)
        ax.set_title(f"{k}   (n_b={load_curve(npz, tag, bl[0])['numel']:,})",
                     fontsize=9, loc="left")
        ax.set_xscale("log"); ax.set_yscale("log")
        # 固定十进位刻度，否则 log 轴的 minor 标签会互相压住（默认会挤成一团）
        ax.xaxis.set_major_locator(FixedLocator([1e0, 1e2, 1e4, 1e6, 1e8]))
        ax.grid(True, which="both", color="#d4d4d4", alpha=0.5, lw=0.4)
        ax.tick_params(labelsize=7)
    max_depth = max(len(v) for v in per_kind.values()) - 1
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_depth))
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cb.set_label("layer depth", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"{CURVE_TITLE.get(tag, tag)} — by tensor type (color = depth)",
                 fontsize=11)
    fig.supxlabel("eigenvalue index (rank)", fontsize=9)
    fig.supylabel("eigenvalue", fontsize=9)
    # ⚠ 不能用 tight_layout：与 colorbar 的 ax=[...] 布局冲突（会警告并错位）
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved:", outpath, flush=True)


def fig_grid(npz, tag, blocks, outpath):
    k = len(blocks)
    ncol = min(5, k)
    nrow = (k + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, b in enumerate(blocks):
        cur = load_curve(npz, tag, b)
        if cur is None:
            continue
        ax = axes[i // ncol][i % ncol]
        ax.axis("on")
        x, y = combined_positive(cur)
        ax.plot(x, y, color="#0072B2", lw=1.2)
        g, mid, lo, hi = cur["g"], cur["mid"], cur["lo"], cur["hi"]
        q = np.isfinite(g) & np.isfinite(mid) & np.isfinite(lo) & np.isfinite(hi) & (g > 0)
        ax.fill_betweenx(g[q], lo[q], hi[q], color="#0072B2", alpha=0.12, lw=0)
        s = summarize(cur["eigs"])
        ax.set_title(f"{b}  n_b={cur['numel']:,}\nλmax={s['lam_max']:.3g}",
                     fontsize=8, loc="left")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator([1e0, 1e2, 1e4, 1e6]))
        ax.grid(True, which="both", color="#d4d4d4", alpha=0.5, lw=0.4)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{CURVE_TITLE.get(tag, tag)} — per-block spectra", fontsize=11)
    fig.supxlabel("eigenvalue index (rank)", fontsize=9)
    fig.supylabel("eigenvalue", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved:", outpath, flush=True)


def _layer_of(name):
    """块名 → 层号；embd/head 等无层号的返回 None。"""
    if name.startswith("layer") and name[5:7].isdigit():
        return int(name[5:7])
    return None


def fig_summary(npz, tag, blocks, outpath):
    """逐块摘要。块细到张量粒度时，**横轴用层深、每个张量类型一条线**，
    直接回答「同一个张量随深度怎么变」。

    ⚠ 别用「按块序号排 + 每隔 k 个打标签」：每层 6 个张量，stride=6 会让所有标签
    恰好落在同一个张量上（首版全打成 mlp_head），读者会以为 74 个点都是 mlp_head。
    """
    rows = []
    for b in blocks:
        cur = load_curve(npz, tag, b)
        if cur is None:
            continue
        s = summarize(cur["eigs"])
        s["name"] = b
        s["kind"] = block_kind(b)
        s["layer"] = _layer_of(b)
        rows.append(s)
    if not rows:
        return

    layered = [r for r in rows if r["layer"] is not None]
    other = [r for r in rows if r["layer"] is None]
    by_depth = bool(layered) and len({r["kind"] for r in layered}) > 1

    fig, axes = plt.subplots(1, 4, figsize=(4.4 * 4, 4.2))
    panels = (
        (axes[0], "lam_max", "λmax", True),
        (axes[1], "trace", "trace(H_bb)", True),
        (axes[2], "eff_rank", "effective rank (Σλ)²/Σλ²", True),
        (axes[3], "frac_neg", "fraction of negative eigenvalues", False),
    )

    if by_depth:
        kinds = sorted({r["kind"] for r in layered},
                       key=lambda k: block_order_key(f"layer00.{k}"))
        for ax, key, ttl, logy in panels:
            for k in kinds:
                pts = sorted([r for r in layered if r["kind"] == k],
                             key=lambda r: r["layer"])
                ax.plot([r["layer"] for r in pts], [r[key] for r in pts],
                        "o-", ms=4, lw=1.1, color=KIND_COLORS.get(k, "#888888"),
                        label=k)
            # embd/head 无层号 → 画成水平参考线
            for r, ls in zip(other, ("--", ":")):
                ax.axhline(r[key], color=KIND_COLORS.get(r["kind"], "#000000"),
                           ls=ls, lw=1.0, alpha=0.75, label=r["name"])
            ax.set_xlabel("layer depth")
            ax.set_title(ttl, fontsize=10, loc="left")
            vals = np.array([r[key] for r in rows], dtype=np.float64)
            if logy and np.all(vals > 0):
                ax.set_yscale("log")
            ax.grid(True, color="#d4d4d4", alpha=0.55, lw=0.45)
        axes[0].legend(frameon=False, fontsize=6.5, ncol=2, loc="best")
    else:
        names = [r["name"] for r in rows]
        cols = [KIND_COLORS.get(r["kind"], "#0072B2") for r in rows]
        xs = np.arange(len(rows))
        for ax, key, ttl, logy in panels:
            vals = np.array([r[key] for r in rows], dtype=np.float64)
            ax.scatter(xs, vals, c=cols, s=18, zorder=3)
            ax.plot(xs, vals, color="#bbbbbb", lw=0.6, zorder=2)
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=80, fontsize=6.5)
            ax.set_title(ttl, fontsize=10, loc="left")
            if logy and np.all(vals > 0):
                ax.set_yscale("log")
            ax.grid(True, color="#d4d4d4", alpha=0.55, lw=0.45)

    fig.suptitle(f"{CURVE_TITLE.get(tag, tag)} — per-block summary", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved:", outpath, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="", help="只画某条曲线（默认 npz 里全部）")
    ap.add_argument("--force", action="store_true", help="允许覆盖已有 png")
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    found = discover(npz)
    if args.tag:
        found = {k: v for k, v in found.items() if k == args.tag}
    if not found:
        raise SystemExit(f"{args.npz} 里没找到任何块曲线（--tag={args.tag}）")
    os.makedirs(args.outdir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.npz))[0]
    for tag, blocks in found.items():
        print(f"[{tag}] {len(blocks)} 块: {', '.join(blocks)}", flush=True)
        for kind, fn in (("overlay", fig_overlay), ("bykind", fig_by_kind),
                         ("grid", fig_grid), ("summary", fig_summary)):
            out = os.path.join(args.outdir, f"{stem}_{tag}_{kind}.png")
            # 不覆盖既有产出
            if os.path.exists(out) and not args.force:
                print(f"  跳过（已存在，加 --force 才覆盖）: {out}", flush=True)
                continue
            fn(npz, tag, blocks, out)


if __name__ == "__main__":
    main()
