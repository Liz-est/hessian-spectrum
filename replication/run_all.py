"""
run_all.py
==========
Run every replication preset, one per GPU, in parallel subprocesses.

Each run is single-GPU (the problem is one dim x dim matrix; no DDP), so with
8 presets and 8 GPUs everything runs concurrently.  If there are more presets
than GPUs, runs are dispatched to GPUs as they free up.

Usage
-----
    python run_all.py                 # all presets
    python run_all.py adam            # only presets whose name contains "adam"
    python run_all.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

import torch

from presets import PRESETS

REPL_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", default="",
                    help="only run presets whose name contains one of these "
                         "comma-separated substrings")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    filts = [f for f in args.filter.split(",") if f]
    names = [n for n in PRESETS
             if (not filts or any(f in n for f in filts)) and "smoke" not in n]
    if not names:
        sys.exit(f"no preset matches '{args.filter}'")
    n_gpu = max(1, torch.cuda.device_count())
    print(f"[run_all] {len(names)} runs on {n_gpu} GPU(s):")
    for n in names:
        print(f"  {n}")
    if args.dry_run:
        return

    pending = list(names)
    running = {}                       # gpu -> (name, Popen)
    failed = []
    while pending or running:
        # reap finished
        for gpu in list(running):
            name, p = running[gpu]
            ret = p.poll()
            if ret is None:
                continue
            status = "ok" if ret == 0 else f"FAILED (exit {ret})"
            print(f"[run_all] {name} on gpu{gpu}: {status}", flush=True)
            if ret != 0:
                failed.append(name)
            del running[gpu]
        # dispatch
        for gpu in range(n_gpu):
            if gpu in running or not pending:
                continue
            name = pending.pop(0)
            log = open(os.path.join(REPL_ROOT, "runs_log_" + name + ".txt"), "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            p = subprocess.Popen(
                [sys.executable, "-u", os.path.join(REPL_ROOT, "train.py"),
                 name, "--device=cuda:0" if torch.cuda.is_available()
                 else "--device=cpu"],
                stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPL_ROOT)
            running[gpu] = (name, p)
            print(f"[run_all] launched {name} on gpu{gpu} (pid {p.pid})",
                  flush=True)
        time.sleep(5)

    if failed:
        sys.exit(f"[run_all] {len(failed)} run(s) failed: {failed}")
    print("[run_all] all runs finished ok")


if __name__ == "__main__":
    main()
