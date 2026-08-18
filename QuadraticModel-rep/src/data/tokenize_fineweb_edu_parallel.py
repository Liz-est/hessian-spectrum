"""
并行版干净分词：FineWeb-Edu 原始 parquet → bpe_8192 → uint16 shard，再按序拼接。
与串行版 tokenize_fineweb_edu.py 完全等价的输出（同 tokenizer / 同 <eot>(id=0)
每文档追加 / 同 train=files[:-1] val=files[-1:] 划分），只是把 14 个 parquet
分派给多进程并行 encode，吃满 SCO 节点 32 vCPU。

流程：
  1. 每个 parquet 由一个 worker 进程独立分词 → 写 shard_{split}_{idx}.bin
     （worker 内 encode_batch 走 tokenizers 的 rayon 多线程；进程数×每进程线程数≈核数）
  2. 主进程按原始文件顺序把 train shard 拼成 train.bin、val shard 拼成 val.bin
     （拼接顺序 = sorted(parquet) 顺序，与串行版逐字节一致）
用法：python tokenize_fineweb_edu_parallel.py [--workers N] [--rayon T]
"""
import os, sys, glob, time, argparse
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool

PARQUET_DIR = "/data/250010020/hessian-spectrum/data/fineweb_edu_10BT_parquet/sample/10BT"
OUT_DIR = "/data/250010020/hessian-spectrum/data/fineweb_edu_bpe8192_clean"
TOKENIZER = os.environ.get("HS_BPE_JSON",
    "/data/250010020/hessian-spectrum/QuadraticModel/tokenizers/bpe_8192.json")
EOT = 0
ROWGROUP_BATCH = 4
SHARD_DIR = os.path.join(OUT_DIR, "_shards")

_TK = None   # 每 worker 进程一份 tokenizer（fork 后惰性初始化）


def _init_worker(rayon_threads):
    # 限制每进程 rayon 线程数，避免多进程 × 多线程超订
    os.environ["RAYON_NUM_THREADS"] = str(rayon_threads)


def _get_tk():
    global _TK
    if _TK is None:
        from tokenizers import Tokenizer
        _TK = Tokenizer.from_file(TOKENIZER)
        assert _TK.token_to_id("<eot>") == EOT
    return _TK


def tokenize_one(task):
    """分词单个 parquet → 独立 shard 文件，返回 (shard_path, ntok, seconds)。"""
    split, idx, path = task
    tk = _get_tk()
    shard_path = os.path.join(SHARD_DIR, f"shard_{split}_{idx:03d}.bin")
    t0 = time.time(); written = 0
    pf = pq.ParquetFile(path)
    nrg = pf.num_row_groups
    with open(shard_path, "wb") as out_f:
        for s in range(0, nrg, ROWGROUP_BATCH):
            groups = list(range(s, min(s + ROWGROUP_BATCH, nrg)))
            tbl = pf.read_row_groups(groups, columns=["text"])
            texts = tbl["text"].to_pylist()
            encs = tk.encode_batch(texts)
            parts = []
            for e in encs:
                parts.append(np.asarray(e.ids, dtype=np.uint16))
                parts.append(np.array([EOT], dtype=np.uint16))
            arr = np.concatenate(parts)
            arr.tofile(out_f)
            written += arr.size
            del tbl, texts, encs, parts, arr
    return shard_path, written, time.time() - t0


def concat_shards(shard_paths, out_path):
    """按给定顺序把 shard 拼接为最终 .bin（流式，不全量载入）。"""
    total = 0
    with open(out_path, "wb") as out_f:
        for sp in shard_paths:
            with open(sp, "rb") as f:
                while True:
                    chunk = f.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
            total += os.path.getsize(sp)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7,
                    help="并行进程数（× rayon 线程 ≈ 核数）")
    ap.add_argument("--rayon", type=int, default=4, help="每进程 rayon 线程数")
    ap.add_argument("--keep_shards", action="store_true", help="拼接后保留 shard")
    args = ap.parse_args()

    os.makedirs(SHARD_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
    assert files, f"无 parquet：{PARQUET_DIR}"
    train_files, val_files = files[:-1], files[-1:]
    print(f"train parquet={len(train_files)}  val parquet={len(val_files)}  "
          f"workers={args.workers} rayon={args.rayon}", flush=True)

    tasks = ([("train", i, fp) for i, fp in enumerate(train_files)] +
             [("val", i, fp) for i, fp in enumerate(val_files)])

    t0 = time.time()
    results = {}
    with Pool(processes=args.workers, initializer=_init_worker,
              initargs=(args.rayon,)) as pool:
        for shard_path, ntok, secs in pool.imap_unordered(tokenize_one, tasks):
            results[shard_path] = ntok
            print(f"  完成 {os.path.basename(shard_path)}  +{ntok:,} tok  "
                  f"{secs:.0f}s  ({ntok/secs/1e6:.2f} Mtok/s)  累计耗时 {time.time()-t0:.0f}s",
                  flush=True)

    # 按原始顺序拼接
    for split, flist in [("train", train_files), ("val", val_files)]:
        n = len(flist)
        shard_paths = [os.path.join(SHARD_DIR, f"shard_{split}_{i:03d}.bin") for i in range(n)]
        out_path = os.path.join(OUT_DIR, f"{split}.bin")
        nbytes = concat_shards(shard_paths, out_path)
        ntok = sum(results[sp] for sp in shard_paths)
        print(f"[{split}] 拼接 {n} shard -> {out_path}  {ntok:,} tok ({nbytes/1e9:.2f} GB)",
              flush=True)

    if not args.keep_shards:
        for sp in glob.glob(os.path.join(SHARD_DIR, "*.bin")):
            os.remove(sp)
        os.rmdir(SHARD_DIR)
    print(f"全部完成，总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
