#!/usr/bin/env python3
"""Submit SCO jobs that serially compute preconditioned-Hessian block
heterogeneity for a GROUP of (run×layer) combos, then plot the group's 2D
trajectories. One job per group (each uses 1 GPU of an 8xH100 node).

Groups:
  frz_embd   -- the original 3 runs (embedding frozen, lm_head trained)
  frz_lmhead -- the new 3 runs (lm_head frozen, embedding trained)

Usage:
    python3 submit_sco_precond.py --group frz_embd   --yes
    python3 submit_sco_precond.py --group frz_lmhead --yes
    python3 submit_sco_precond.py --group all        --yes   # both, 2 jobs
"""
import argparse
import subprocess
import sys
import time

SCO = "/root/.sco/bin/sco"
PROFILE = "zhanglixian-g"
WORKSPACE_NAME = "p10-intelligent-adaptation-and-optimization-for-domestic-ai"
AEC2_NAME = "share-cluster"
CONTAINER_IMAGE_URL = (
    "registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/"
    "zkx-ssh-install-g:main-20260515065803"
)
TRAINING_FRAMEWORK = "pytorch"
WORKER_NODES = 1
WORKER_SPEC = "n6ls.iu.i40.8.32c512g"
STORAGE_MOUNT = "01995892-d478-76d8-aec7-13fd8284477e:/data"

USER_DATA = "/data/250010020"
REPO_ROOT = f"{USER_DATA}/hessian-spectrum"
WORK_DIR = f"{REPO_ROOT}/toy_models"

CONDA_ENV_PATH = f"{USER_DATA}/miniconda3/envs/nanogpt"
ENV_PYTHON = f"{CONDA_ENV_PATH}/bin/python"

LAYERS = ["lm_head", "embedding"]

# (run_dir_basename, optim_as_trained) grouped by which weight was frozen
GROUPS = {
    "frz_embd": [
        ("REP-mserep-pos0-frz_embd-fullbs-sgd-lr0p048-imb-initG02-nobias", "sgd"),
        ("REP-mserep-pos0-frz_embd-fullbs-adam-lr3e-6-imb-initG02-nobias", "adam"),
        ("REP-muon-lr6e-5-G02-mom0", "muon"),
    ],
    "frz_lmhead": [
        ("REP1-frz_lmhead-sgd-lr0p01", "sgd"),
        ("REP1-frz_lmhead-adam-lr2e-5-G02", "adam"),
        ("REP1-frz_lmhead-muon-lr1e-4-G02-mom0", "muon"),
    ],
}


def build_command(runs, group, only_optims=None):
    """only_optims: if set, only run compute for these optimizers (others reuse
    existing summaries). Plot always uses all runs in the group."""
    compute = [
        f"{ENV_PYTHON} -u compute_precond_hessian.py runs/{run} "
        f"--layer {layer} --optim {optim} --device cuda"
        for run, optim in runs for layer in LAYERS
        if (only_optims is None or optim in only_optims)
    ]
    plot = (f"{ENV_PYTHON} -u plot_precond_2d.py "
            + " ".join(f"runs/{r}" for r, _ in runs)
            + f" --out_dir files/precond_2d_{group}")
    return (f"cd {WORK_DIR} && export PATH={CONDA_ENV_PATH}/bin:$PATH && "
            f"export CUDA_VISIBLE_DEVICES=0 && "
            + " && ".join(compute + [plot]))


def submit(job_name, command):
    cmd = [
        SCO, "--profile", PROFILE, "acp", "jobs", "create",
        "--workspace-name", WORKSPACE_NAME,
        "--aec2-name", AEC2_NAME,
        "--job-name", job_name,
        "--container-image-url", CONTAINER_IMAGE_URL,
        "--training-framework", TRAINING_FRAMEWORK,
        "--worker-nodes", str(WORKER_NODES),
        "--worker-spec", WORKER_SPEC,
        "--storage-mount", STORAGE_MOUNT,
        "--command", command,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        print(f"✓ {job_name} submitted")
        if result.stdout:
            print("  " + result.stdout.strip())
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"✗ {job_name} failed")
        if getattr(exc, "stdout", None):
            print(exc.stdout)
        if getattr(exc, "stderr", None):
            print(exc.stderr)
        return False


# Per-group: which optimizers to (re)compute. frz_embd was already computed for
# SGD/Muon (unaffected by the Adam v^{-1/4} fix), so only Adam is recomputed;
# frz_lmhead is brand new -> compute all.
RECOMPUTE = {
    "frz_embd": {"adam"},
    "frz_lmhead": None,   # None = all optimizers
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(GROUPS) + ["all"], default="all")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    groups = list(GROUPS) if args.group == "all" else [args.group]
    print(f"Groups to submit: {groups}")
    for g in groups:
        only = RECOMPUTE.get(g)
        which = "all optims" if only is None else f"only {sorted(only)} (others reuse)"
        print(f"  {g}: {len(GROUPS[g])} runs x {len(LAYERS)} layers, compute {which}")
    if not args.yes:
        if input("Submit? (y/n): ").strip().lower() != "y":
            print("Cancelled.")
            return

    ok = 0
    for g in groups:
        job_name = f"precond-{g}-002"
        cmd = build_command(GROUPS[g], g, only_optims=RECOMPUTE.get(g))
        if submit(job_name, cmd):
            ok += 1
        time.sleep(1.0)
    print(f"\n{ok}/{len(groups)} jobs submitted.")
    if ok < len(groups):
        sys.exit(1)


if __name__ == "__main__":
    main()

