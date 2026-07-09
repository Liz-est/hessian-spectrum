import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def kl_divergence(p, q, eps=1e-12):
    """KL(P||Q)，避免 log(0) 问题"""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))

def js_distance(p, q):
    """手写 JS 距离 (取 sqrt 后的 JS divergence)"""
    m = 0.5 * (p + q)
    js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return np.sqrt(js_div)

def compute_js_matrix(values_layer_json):
    """
    values_layer_json: dict
        {layer_name: [[...], [...], ...]}  # num_v × m
    返回: (names, js_mat)
    """
    names = list(values_layer_json.keys())
    save_names = [name.removeprefix('module.') for name in values_layer_json.keys()]
    n = len(names)
    js_mat = np.zeros((n, n))

    # 转换为概率分布
    prob_dists = {}
    for idx, name in enumerate(names):
        arr = np.array(values_layer_json[name])  # (num_v, m)
        avg = arr.mean(axis=0)
        prob = avg / avg.sum()
        prob_dists[name] = prob

    # 计算两两 JS
    for i in range(n):
        for j in range(i):
            p, q = prob_dists[names[i]], prob_dists[names[j]]
            dist = js_distance(p, q)
            js_mat[i, j] = dist
            js_mat[j, i] = dist  # 对称填充，方便可视化

    return save_names, js_mat

def plot_js_heatmap(names, js_mat, save_dir=".", fname="js_heatmap"):
    os.makedirs(save_dir, exist_ok=True)

    # 下三角 mask
    mask = np.triu(np.ones_like(js_mat, dtype=bool), k=1)

    plt.figure(figsize=(8, 6))
    sns.heatmap(js_mat, xticklabels=names, yticklabels=names,
                mask=mask, cmap="RdBu_r", vmin=0, vmax=3, square=True, 
                cbar_kws={"label": "JS Distance"})

    # 计算均值（只取下三角的非零部分）
    tril_indices = np.tril_indices_from(js_mat, k=-1)
    js_mean = js_mat[tril_indices].mean() if len(tril_indices[0]) > 0 else 0.0

    # 在图下方加文字
    plt.figtext(0.5, -0.05, f"JS = {js_mean:.4f}", ha="center", fontsize=12)
    plt.title("JS Distance Heatmap")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{fname}.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

    # 保存矩阵数据
    np.save(os.path.join(save_dir, f"{fname}.npy"), js_mat)
    print(f"✅ Heatmap saved to {save_path}")
    print(f"✅ Matrix saved to {os.path.join(save_dir, f'{fname}.npy')}")
    print(f"✅ JS mean = {js_mean:.4f}")

if __name__ == "__main__":
    # 假设 values_layer.json 路径
    base_path = "./files/model-resnet18-opt-adamw-lr-0.01-beta1-0.9-beta2-0.999-wd-0.0001-seed-32-dsmode-1k_minibatch_True_bs_1920_m_10_v_1_ckpt_89/"
    json_path = base_path + "values_layer.json"
    with open(json_path, "r") as f:
        values_layer = json.load(f)

    names, js_mat = compute_js_matrix(values_layer)
    plot_js_heatmap(names, js_mat, save_dir=base_path, fname="js_heatmap")
