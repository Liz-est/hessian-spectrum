import os
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt

# 要读取的文件名
METRICS_FILENAME = "last_layer_subset_hessian_condition_module_heads_head_weight.txt"

# 匹配文件夹名里 epoch 的正则：..._ckpt_10 之类
EPOCH_PATTERN = re.compile(r"_ckpt_(\d+)$")

# 匹配科学计数法的浮点数
FLOAT_PATTERN = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"


def collect_hessian_metrics(parent_dir: str | Path) -> Dict[int, Tuple[float, float]]:
    """
    遍历 parent_dir 下所有子文件夹，找到名字以 `_ckpt_{epoch}` 结尾的文件夹，
    读取其中的 metrics txt 文件，解析出：
        - Modified condition number
        - Hessian diagonal energy ratio

    返回:
        dict[epoch] = (modified_condition_number, hessian_diagonal_energy_ratio)
    """
    parent_dir = Path(parent_dir)
    results: Dict[int, Tuple[float, float]] = {}

    # 递归遍历所有子目录
    for dirpath, dirnames, filenames in os.walk(parent_dir):
        dirpath = Path(dirpath)
        dirname = dirpath.name

        m = EPOCH_PATTERN.search(dirname)
        if not m:
            continue  # 不是 _ckpt_{epoch} 结尾的目录，跳过

        epoch = int(m.group(1))
        metrics_path = dirpath / METRICS_FILENAME
        if not metrics_path.is_file():
            # 找不到指定的 txt 文件，跳过
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            text = f.read()

        # 解析 Modified condition number
        mod_match = re.search(
            r"Modified condition number.*?:\s*" + FLOAT_PATTERN,
            text,
            flags=re.IGNORECASE,
        )
        # 解析 Hessian diagonal energy ratio
        ratio_match = re.search(
            r"Hessian diagonal energy ratio.*?:\s*" + FLOAT_PATTERN,
            text,
            flags=re.IGNORECASE,
        )

        if not (mod_match and ratio_match):
            # 某一项没解析到就跳过这个 epoch
            continue

        modified_cond = float(mod_match.group(1))
        diag_ratio = float(ratio_match.group(1))

        results[epoch] = (modified_cond, diag_ratio)

    # 按 epoch 排序后返回
    return dict(sorted(results.items(), key=lambda kv: kv[0]))


def _unpack_metrics(
    metrics: Dict[int, Tuple[float, float]]
) -> Tuple[list[int], list[float], list[float]]:
    """把 {epoch: (modified, ratio)} 拆成三个 list，方便画图"""
    epochs = sorted(metrics.keys())
    indices = [i for i in range(len(epochs))]
    modified = [metrics[e][0] for e in epochs]
    ratios = [metrics[e][1] for e in epochs]
    return epochs, indices, modified, ratios


def plot_two_runs(
    metrics_a: Dict[int, Tuple[float, float]],
    metrics_b: Dict[int, Tuple[float, float]],
    label_a: str = "run A",
    label_b: str = "run B",
):
    """
    将两个 dict 画在同一张图的两行子图上：
      上：modified condition number 折线图
      下：Hessian diagonal energy ratio 折线图
    """
    if not metrics_a or not metrics_b:
        raise ValueError("metrics_a 和 metrics_b 都需要包含至少一个 epoch 数据。")

    epochs_a, indices_a, mod_a, ratio_a = _unpack_metrics(metrics_a)
    epochs_b, indices_b, mod_b, ratio_b = _unpack_metrics(metrics_b)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

    # ---------- 子图 1：Modified condition number ----------
    ax1.plot(indices_a, mod_a, marker="o", linestyle="-", label=label_a)
    ax1.plot(indices_b, mod_b, marker="s", linestyle="-", label=label_b)
    ax1.set_yscale("log")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Modified condition number")
    ax1.set_title("Modified condition number vs. Index")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ---------- 子图 2：Hessian diagonal energy ratio ----------
    ax2.plot(indices_a, ratio_a, marker="o", linestyle="-", label=label_a)
    ax2.plot(indices_b, ratio_b, marker="s", linestyle="-", label=label_b)
#     ax2.set_yscale("log")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Hessian diagonal energy ratio")
    ax2.set_title("Hessian diagonal energy ratio vs. Index")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("/libingheng/hessian-spectrum/vision_models/see.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    
# 获取两个实验的指标字典
metrics_run1 = collect_hessian_metrics("/libingheng/hessian-spectrum/vision_models/files/zero_init_loss")
metrics_run2 = collect_hessian_metrics("/libingheng/hessian-spectrum/vision_models/files/normal_init_loss")

# 画在同一张图上
plot_two_runs(metrics_run1, metrics_run2, label_a="zero_init", label_b="normal_init")

