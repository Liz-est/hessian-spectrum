#!/usr/bin/env python3
"""
Submit one SCO job that runs the vanilla_transformer (vanilla decoder) experiment on
8 GPUs: DDP training, then the 8-GPU-sharded exact per-unit Hessian analysis,
and -- if the resolved config has analyze.slq=True -- the full-parameter Hessian
SLQ pass (analyze_full_spectrum.py), all in a single job.

Usage (from the hessian-spectrum repo):
    python3 submit_sco_vanilla.py            # asks for confirmation
    python3 submit_sco_vanilla.py --yes      # submit without prompt

Notes
-----
* Mirrors submit_sco_simpliest.py (same platform params / env).
* The container only mounts /data, so the repo, conda env, and dataset are all
  referenced by their /data paths (valid both on this dev box and in the job).
* All experiment settings come from the config package (toy_models/config/).
  Pick a preset / override fields via EXP_ARGS below; the SAME tokens go to
  every phase so they agree on run_name / files_name / ckpt schedule.
* Phase 1 (train_vanilla_transformer.py) launches 8-GPU DDP via torchrun and writes one
  checkpoint per tag in the preset's ckpt_fracs under toy_models/runs/<run_name>/.
* Phase 2 (analyze_vanilla.py) launches 8-GPU torchrun; the (checkpoint x layer) work
  items are sharded across the 8 ranks. Rank 0 renders all figures at the end.
* Phase 3 (analyze_full_spectrum.py, only when analyze.slq is on -- set it in
  the preset or pass --analyze.slq=true in EXP_ARGS): full-parameter Hessian
  ESD via SLQ; (checkpoint x probe) Lanczos runs sharded across the 8 ranks,
  output under files/<files_name>/slq/.
* The dataset named by the preset (cfg.data.dataset) must already be built
  under data/ -- the job does not build it.
"""

import argparse
import subprocess
import sys

import config as cfgmod

# ---- SCO binary + profile (do not use the bare `sco` on PATH) ----
SCO = "/root/.sco/bin/sco"
PROFILE = "zhanglixian-g"

# ---- Platform params (reused from the verified team setup) ------------
WORKSPACE_NAME = "p10-intelligent-adaptation-and-optimization-for-domestic-ai"
AEC2_NAME = "share-cluster"
CONTAINER_IMAGE_URL = (
    "registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/"
    "zkx-ssh-install-g:main-20260515065803"
)
TRAINING_FRAMEWORK = "pytorch"
WORKER_NODES = 1
WORKER_SPEC = "n6ls.iu.i40.8.32c512g"   # 8x H100
STORAGE_MOUNT = "01995892-d478-76d8-aec7-13fd8284477e:/data"

# ===== EDIT THESE FOR YOUR EXPERIMENT ==================================
USER_DATA = "/data/" + "250010020"   # this dev box's /data/250010020 == shared storage /data/<user-id>
REPO_ROOT = f"{USER_DATA}/hessian-spectrum"
WORK_DIR = f"{REPO_ROOT}/toy_models"
JOB_NAME = "vanilla-hessian-003"    # change per run, keep unique

# nanogpt env python by absolute path (no conda activation needed).
CONDA_ENV_PATH = f"{USER_DATA}/miniconda3/envs/nanogpt"
ENV_PYTHON = f"{CONDA_ENV_PATH}/bin/python"

NPROC_PER_NODE = 8

# Config selection + overrides. A bare token picks a preset (config/presets.py);
# --group.key=value overrides one field. The SAME tokens are applied to every
# phase so they all resolve the same run_name / files_name / ckpt schedule.
#   e.g. EXP_ARGS = "imbalance_s1_adamw"
#        EXP_ARGS = "imbalance_s1_adamw --analyze.slq=true"
EXP_ARGS = "mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-init1G"       # applied to ALL phases

# resolve the config locally to read the analyze.slq switch (and echo paths);
# the job re-resolves the same tokens on the cluster.
_cfg = cfgmod.apply_overrides(cfgmod.load(EXP_ARGS.split()[0]), EXP_ARGS.split()[1:])

def _launch(script):
    return (f"{ENV_PYTHON} -u -m torch.distributed.run --standalone "
            f"--nproc_per_node={NPROC_PER_NODE} {script} {EXP_ARGS}")

# run inside the container: cd into toy_models/, put the env python on PATH so
# torchrun-spawned workers resolve the same interpreter, then run the phases.
PHASES = ["train_vanilla_transformer.py", "analyze_vanilla.py"]
if _cfg.analyze.slq:
    PHASES.append("analyze_full_spectrum.py")

# per-frequency-group class-loss curves from the saved ckpts (single process,
# seconds on GPU). Runs right after training so the figure exists even if the
# Hessian phase dies; dual-stream-only, but the script skips non-dual-stream
# datasets (fineweb) gracefully with exit code 0, so it is safe to always run.
_EVAL_BY_FREQ = (f"{ENV_PYTHON} -u eval_ckpts_val_by_freq.py "
                 f"{_cfg.train.run_name} --batch_size=64")

COMMAND = (
    f"cd {WORK_DIR} && "
    f"export PATH={CONDA_ENV_PATH}/bin:$PATH && "
    + " && ".join([_launch(PHASES[0]), _EVAL_BY_FREQ]
                  + [_launch(s) for s in PHASES[1:]])
)
# =======================================================================


def submit_job() -> bool:
    cmd = [
        SCO, "--profile", PROFILE,
        "acp", "jobs", "create",
        "--workspace-name", WORKSPACE_NAME,
        "--aec2-name", AEC2_NAME,
        "--job-name", JOB_NAME,
        "--container-image-url", CONTAINER_IMAGE_URL,
        "--training-framework", TRAINING_FRAMEWORK,
        "--worker-nodes", str(WORKER_NODES),
        "--worker-spec", WORKER_SPEC,
        "--storage-mount", STORAGE_MOUNT,
        "--command", COMMAND,
    ]
    print(f"Submitting job: {JOB_NAME}")
    print(f"Command: {COMMAND}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Job {JOB_NAME} submitted.")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Job {JOB_NAME} failed to submit.")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="skip confirmation")
    args = parser.parse_args()

    print(f"Job name:   {JOB_NAME}")
    print(f"Worker spec:{WORKER_SPEC}")
    print(f"Work dir:   {WORK_DIR}")
    print(f"Config:     {EXP_ARGS}")
    print(f"Run dir:    runs/{_cfg.train.run_name}")
    print(f"Files dir:  files/{_cfg.analyze.files_name}")
    print(f"DDP:        torchrun, {NPROC_PER_NODE} GPUs x {WORKER_NODES} node(s)")
    slq = "  ->  (3) full-parameter SLQ" if _cfg.analyze.slq else ""
    print(f"Phases:     (1) DDP train  ->  (1b) class-loss-by-freq eval"
          f"  ->  (2) per-unit Hessian analysis{slq}")
    if not args.yes:
        if input("Continue? (y/n): ").strip().lower() != "y":
            print("Cancelled.")
            return
    if not submit_job():
        sys.exit(1)


if __name__ == "__main__":
    main()
