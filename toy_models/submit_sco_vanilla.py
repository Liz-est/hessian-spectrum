#!/usr/bin/env python3
"""
Submit one SCO job that runs the vanilla_transformer (vanilla decoder) experiment on
8 GPUs: first DDP training, then the 8-GPU-sharded exact per-unit Hessian
analysis, all in a single job.

Usage (from the hessian-spectrum repo):
    python3 submit_sco_vanilla.py            # asks for confirmation
    python3 submit_sco_vanilla.py --yes      # submit without prompt

Notes
-----
* Mirrors language_models/submit_sco.py (same platform params / env).
* The container only mounts /data, so the repo, conda env, and dataset are all
  referenced by their /data paths (valid both on this dev box and in the job).
* All experiment settings come from the config package (toy_models/config/).
  Pick a preset / override fields via TRAIN_ARGS and ANALYZE_ARGS below; the
  SAME tokens must go to both phases so they agree on run_name / ckpt schedule.
* Phase 1 (train_vanilla_transformer.py) launches 8-GPU DDP via torchrun and writes one
  checkpoint per tag in the preset's ckpt_fracs under toy_models/runs/<run_name>/.
* Phase 2 (analyze_vanilla.py) launches 8-GPU torchrun; the (checkpoint x layer) work
  items are sharded across the 8 ranks. Rank 0 renders all figures at the end.
* The dataset named by the preset (cfg.data.dataset) must already be built
  under data/ -- the job does not build it.
"""

import argparse
import subprocess
import sys

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
# --group.key=value overrides one field. Keep TRAIN/ANALYZE in sync so both
# phases resolve the same run_name and checkpoint schedule.
#   e.g. EXP_ARGS = "imbalance_s1_adamw"
#        EXP_ARGS = "--optim.name=adamw --lr.learning_rate=3e-4"
EXP_ARGS = "layer5-fineweb10B-sgd"       # applied to BOTH phases
TRAIN_ARGS = EXP_ARGS       # e.g. EXP_ARGS + " --train.max_iters=8000"
# full-vocab lm_head/embedding analysis: add " --analyze.max_classes=1024 --analyze.max_tokens=1024"
ANALYZE_ARGS = EXP_ARGS
    
TRAIN_LAUNCH = (
    f"{ENV_PYTHON} -u -m torch.distributed.run --standalone "
    f"--nproc_per_node={NPROC_PER_NODE} train_vanilla_transformer.py {TRAIN_ARGS}"
)
ANALYZE_LAUNCH = (
    f"{ENV_PYTHON} -u -m torch.distributed.run --standalone "
    f"--nproc_per_node={NPROC_PER_NODE} analyze_vanilla.py {ANALYZE_ARGS}"
)

# run inside the container: cd into toy_models/, put the env python on PATH so
# torchrun-spawned workers resolve the same interpreter, then train then analyze.
COMMAND = (
    f"cd {WORK_DIR} && "
    f"export PATH={CONDA_ENV_PATH}/bin:$PATH && "
    f"{TRAIN_LAUNCH} && "
    f"{ANALYZE_LAUNCH}"
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
    print("Phases:     (1) DDP train  ->  (2) 8-GPU-sharded Hessian analysis")
    if not args.yes:
        if input("Continue? (y/n): ").strip().lower() != "y":
            print("Cancelled.")
            return
    if not submit_job():
        sys.exit(1)


if __name__ == "__main__":
    main()
