"""路径中枢：本仓库内所有数据/产物路径的唯一来源。

约定（2026-08 重构）：
- 数据集与论文谱缓存是只读共享资源，默认指合作者主目录 /data/250010020/hessian-spectrum/，
  用 HS_DATA_DIR / HS_CACHE_NPZ 环境变量覆盖。
- checkpoint / outputs / figures 等产物一律在本仓库根下（分支重训自产，与主目录互不干扰）。
- 对比合作者结果时用 HS_PEER_REP 指到对方的 QuadraticModel-rep（默认主目录那份）。
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_MAIN = "/data/250010020/hessian-spectrum"

# 只读共享资源（主目录）
DATA_DIR = Path(os.environ.get(
    "HS_DATA_DIR", f"{_MAIN}/data/fineweb_edu_100B_parquet/sample/100BT"))
CACHE_NPZ = Path(os.environ.get(
    "HS_CACHE_NPZ", f"{_MAIN}/QuadraticModel/analysis/data/cache/spectrum_3x3.npz"))

# 对比对象（合作者的 rep 产物）
PEER_REP = Path(os.environ.get("HS_PEER_REP", f"{_MAIN}/QuadraticModel-rep"))

# 本仓库资源与产物
TOKENIZER_DIR = REPO_ROOT / "tokenizers"
CKPT_DIR = REPO_ROOT / "checkpoints_b64"
CKPT_DIR_MUON = REPO_ROOT / "checkpoints_b64_muon"
OUT_DIR = REPO_ROOT / "outputs"
FIG_DIR = REPO_ROOT / "figures"
TEST_OUT_DIR = REPO_ROOT / "test_outputs"


def ensure(p):
    """mkdir -p 后返回 Path，产物目录写入前调用。"""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


if __name__ == "__main__":
    for name in ("REPO_ROOT", "DATA_DIR", "CACHE_NPZ", "PEER_REP",
                 "TOKENIZER_DIR", "CKPT_DIR", "OUT_DIR"):
        p = globals()[name]
        print(f"{name:14s} = {p}  [{'exists' if Path(p).exists() else 'MISSING'}]")
