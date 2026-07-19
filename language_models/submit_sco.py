#!/usr/bin/env python3
"""
Submit one SCO job for the language_models GPT-2 hessian-spectrum experiment.

Edit the marked sections below, then run (from the hessian-spectrum repo):
    python3 submit_sco.py            # asks for confirmation
    python3 submit_sco.py --yes      # submit without prompt

Notes
-----
* The job first creates the `gpt2` conda env from environment.yml if it does
  not exist yet (idempotent), then activates it and launches training.
* Dataset: train_gpt2.py reads ../../data_construction/data/<dataset> relative
  to the working directory (language_models/). Make sure the dataset is built
  BEFORE submitting -- the job does not build it.
* Launches 8-GPU DDP via torchrun (train_gpt2.py has been fixed to use
  DistributedDataParallel). The Hessian spectrum pass at startup runs on
  rank 0 only; other ranks wait at a barrier.
"""

import argparse
import subprocess
import sys

# ---- SCO binary + your profile (do not use the bare `sco` on PATH) ----
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
REPO_ROOT = "/data/250010020/hessian-spectrum"
WORK_DIR = f"{REPO_ROOT}/language_models"
JOB_NAME = "zlx-gpt2-hessian-001"        # change per run, keep unique

CONDA_SH = "/data/250010020/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV = "gpt2"                       # created from environment.yml on first run

# 8-GPU single-node DDP. gradient_accumulation_steps in the config (40)
# must stay divisible by NPROC_PER_NODE (asserted in train_gpt2.py).
NPROC_PER_NODE = 8

TRAIN_ARGS = "config/train_gpt2_small.py --dataset=synth_uniform_balanced"

LAUNCH = (
    f"torchrun --standalone --nproc_per_node={NPROC_PER_NODE} "
    f"train_gpt2.py {TRAIN_ARGS}"
)

# The command that runs inside the container: activate (creating if needed)
# the gpt2 conda env, then launch training from language_models/ so that the
# relative paths (configurator.py, config/, ../../data_construction) resolve.
COMMAND = (
    f"cd {WORK_DIR} && "
    f"source {CONDA_SH} && "
    f"(conda env list | grep -qw {CONDA_ENV} || "
    f"conda env create -n {CONDA_ENV} -f environment.yml) && "
    f"conda activate {CONDA_ENV} && "
    f"{LAUNCH}"
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
    print(f"DDP:        torchrun, {NPROC_PER_NODE} GPUs x {WORKER_NODES} node(s)")
    if not args.yes:
        if input("Continue? (y/n): ").strip().lower() != "y":
            print("Cancelled.")
            return
    if not submit_job():
        sys.exit(1)


if __name__ == "__main__":
    main()
