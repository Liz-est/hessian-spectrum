#!/usr/bin/env python3
"""
submit_sco.py
=============
Submit ONE SCO job that runs ALL replication presets (RotatedMatrixBigramProblem,
dim=10000) via run_all.py -- each run is single-GPU, and with 8 presets on an
8x H100 node they all execute in parallel.

Usage (from the repo, or anywhere):
    python3 replication/submit_sco.py            # asks for confirmation
    python3 replication/submit_sco.py --yes      # submit without prompt
    python3 replication/submit_sco.py --filter adam   # only matching presets

Platform params mirror toy_models/submit_sco_vanilla.py (verified team setup).
Outputs land under replication/runs/<preset>/ on the shared /data mount.
"""

import argparse
import subprocess
import sys
import time

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

USER_DATA = "/data/" + "250010020"
REPO_ROOT = f"{USER_DATA}/hessian-spectrum"
WORK_DIR = f"{REPO_ROOT}/replication"
CONDA_ENV_PATH = f"{USER_DATA}/miniconda3/envs/nanogpt"
ENV_PYTHON = f"{CONDA_ENV_PATH}/bin/python"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="skip confirmation")
    parser.add_argument("--filter", default="",
                        help="only run presets whose name contains this")
    args = parser.parse_args()

    ftag = args.filter.replace(",", "_").replace("/", "_")[:30]
    job_name = ("rep-" + (ftag or "all") + "-"
                + time.strftime("%m%d%H%M"))

    command = (
        f"cd {WORK_DIR} && "
        f"export PATH={CONDA_ENV_PATH}/bin:$PATH && "
        f"{ENV_PYTHON} -u run_all.py {args.filter} && "
        f"{ENV_PYTHON} -u plot_compare.py"
    )

    print(f"Job name:   {job_name}")
    print(f"Worker spec:{WORKER_SPEC}")
    print(f"Work dir:   {WORK_DIR}")
    print(f"Command:    {command}")
    if not args.yes:
        if input("Continue? (y/n): ").strip().lower() != "y":
            print("Cancelled.")
            return

    cmd = [
        SCO, "--profile", PROFILE,
        "acp", "jobs", "create",
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
    print(f"Submitting job: {job_name}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Job {job_name} submitted.")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as exc:
        print(f"Job {job_name} failed to submit.")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
