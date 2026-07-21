#!/usr/bin/env python3
"""
Submit one SCO job for the language_models GPT-2 hessian-spectrum experiment.

Edit the marked sections below, then run (from the hessian-spectrum repo):
    python3 submit_sco.py            # asks for confirmation
    python3 submit_sco.py --yes      # submit without prompt

Notes
-----
* The job uses the existing `nanogpt` env from the shared /data storage by
  invoking its python directly (no conda activation, no env creation).
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
# NOTE: the job container only mounts /data. This dev machine's /data/250010020 is
# actually the shared storage's /data/250010020 directory, so everything
# (repo, conda env, dataset) is referenced via its /data path, which is
# valid both here and inside the container.
USER_DATA = "/data/" + "250010020"   # this dev box's /data/250010020 == /data/<user-id> on shared storage
REPO_ROOT = f"{USER_DATA}/hessian-spectrum"
WORK_DIR = f"{REPO_ROOT}/language_models"
JOB_NAME = "gpt2-hessian-002"        # change per run, keep unique

# No conda activation: miniconda3/ on shared storage is a bare cloned-envs
# tree (no etc/profile.d/conda.sh), and anaconda3/'s conda.sh hardcodes
# /mnt/afs/250010020 paths that don't exist inside the container. The env
# has no activate.d hooks, so invoking its python directly is equivalent.
CONDA_ENV_PATH = f"{USER_DATA}/miniconda3/envs/nanogpt"
ENV_PYTHON = f"{CONDA_ENV_PATH}/bin/python"

# 8-GPU single-node DDP. gradient_accumulation_steps in the config (40)
# must stay divisible by NPROC_PER_NODE (asserted in train_gpt2.py).
NPROC_PER_NODE = 8

TRAIN_ARGS = "config/train_gpt2_small.py --dataset=synth_uniform_balanced"

LAUNCH = (
    f"{ENV_PYTHON} -u -m torch.distributed.run --standalone "
    f"--nproc_per_node={NPROC_PER_NODE} train_gpt2.py {TRAIN_ARGS}"
)

# The command that runs inside the container: run the nanogpt env's python
# by absolute path (PATH prepended so torchrun-spawned workers resolve the
# same interpreter), then launch training from language_models/ so the
# relative paths (configurator.py, config/, ../data_construction) resolve.
COMMAND = (
    f"cd {WORK_DIR} && "
    f"export PATH={CONDA_ENV_PATH}/bin:$PATH && "
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
