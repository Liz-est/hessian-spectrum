#!/usr/bin/env python3
"""
Submit one 8-GPU SCO job that runs the FULL-BATCH GD-vs-Adam toy experiments
(train_fullbatch.py) in parallel, one experiment per GPU.

Why this shape
--------------
* The SCO worker spec is a whole 8-GPU node, but train_fullbatch.py is
  deliberately SINGLE-PROCESS (no DDP: the full batch fits one device and
  gradient accumulation already reproduces the exact full-batch gradient).
  So instead of torchrun, each entry in RUNS below is pinned to its own GPU
  via CUDA_VISIBLE_DEVICES and they all run concurrently inside one job.
* Up to 8 entries fit one job -- add lr-sweep variants to fill idle GPUs
  (give each a unique --train.run_name so outputs don't collide).

Usage (from toy_models/):
    python3 submit_sco_fullbatch.py            # asks for confirmation
    python3 submit_sco_fullbatch.py --yes      # submit without prompt

Notes
-----
* Mirrors submit_sco_simpliest.py (same platform params / env python).
* The container only mounts /data, so repo, conda env, and dataset are all
  referenced by /data paths (valid both on this dev box and in the job).
* Per-run stdout goes to logs/<JOB_NAME>/<run>.log inside toy_models/; a
  `tail -f` on all logs multiplexes them into the job's main log so
  `sco acp jobs stream-logs` still shows live progress from every run.
* The actual launcher is WRITTEN TO logs/<JOB_NAME>/launch.sh at submit time
  and the SCO --command just runs `bash <that path>`. Do NOT inline the
  launches into --command: `A && B & C &` parses as `(A && B) & (C) &`, so the
  `cd`/`export PATH` get swallowed by the first background subshell and every
  later run starts in the wrong directory (this silently ate the 2nd run once).
  Keeping it in a file also records exactly what each job ran.
* The job exits non-zero if ANY run fails (each run's exit code is checked).
* The dataset named by each preset (cfg.data.dataset) must already be built
  under data_construction/data/ -- the job does not build it.
"""

import argparse
import os
import re
import subprocess
import sys

# ---- SCO binary + profile (do not use the bare `sco` on PATH) ----
# NOTE: this box only has credentials for sunruoyu-g
# (/root/.config/sco/profiles/sunruoyu-g.toml, also the active_profile).
# The older toy_models submitters name zhanglixian-g, whose profile is gone --
# /root is ephemeral and was not restored, so that profile fails with
# "failed to get SCO_ACCESS_KEY_ID".
SCO = "/root/.sco/bin/sco"
PROFILE = "sunruoyu-g"

# ---- Platform params (reused from the verified team setup) ------------
WORKSPACE_NAME = "p10-intelligent-adaptation-and-optimization-for-domestic-ai"
AEC2_NAME = "share-cluster"
CONTAINER_IMAGE_URL = (
    "registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/"
    "zkx-ssh-install-g:main-20260515065803"
)
TRAINING_FRAMEWORK = "pytorch"
WORKER_NODES = 1
WORKER_SPEC = "n6ls.iu.i40.8.32c512g"   # 8x H100 (whole node; runs are pinned per GPU)
STORAGE_MOUNT = "01995892-d478-76d8-aec7-13fd8284477e:/data"
# The sunruoyu-g submitters in llm-optimizer-benchmark/sco pass these two;
# keep them so the quota resolves the same way for this profile.
PRIORITY = "NORMAL"
QUOTA_TYPE = "reserved"
N_GPUS = 8

# ===== EDIT THESE FOR YOUR EXPERIMENT ==================================
# This repo checkout (under the shared /data mount, so the same path is
# valid inside the job container).
REPO_ROOT = "/data/sunruoyu/wangsenmiao/Data-Imbalance-Hessian/hessian-spectrum-xinlurepo-senmiao"
WORK_DIR = f"{REPO_ROOT}/toy_models"
JOB_NAME = "fullbatch-gd-vs-adam-002"   # change per run, keep unique

# Python environment, by absolute path (no conda activation needed).
# Uses this user's own miniconda3 -- the older toy_models submitters point at
# /data/250010020/miniconda3, another user's dir, which is not readable here.
# `pc` is the only local env with all three of torch / numpy / matplotlib;
# nanogpt and llm-opt have torch+numpy but NO matplotlib, and
# train_fullbatch.py imports matplotlib at module level, so they crash on
# import. Verified 2026-07-30: pc = torch 2.7.0+cu126, numpy 2.2.6, mpl 3.10.7.
CONDA_ENV_PATH = "/data/sunruoyu/wangsenmiao/miniconda3/envs/pc"
ENV_PYTHON = f"{CONDA_ENV_PATH}/bin/python"

# One entry per GPU (index == CUDA_VISIBLE_DEVICES). Each entry is the CLI
# for train_fullbatch.py: "<preset> [--group.key=value ...]".
# IMPORTANT: entries sharing a preset MUST override --train.run_name (and
# --analyze.files_name if analyzed later) or they write into the same runs/ dir.
RUNS = [
    "fullbatch-mse0-shuffled-gd",
    "fullbatch-mse0-shuffled-adam",
    # --- lr sweep examples to fill the remaining GPUs -----------------
    # "fullbatch-mse0-shuffled-gd --lr.learning_rate=3e-3"
    #     " --train.run_name=fullbatch-mse0-shuffled-gd-lr3e-3",
    # "fullbatch-mse0-shuffled-adam --lr.learning_rate=3e-3"
    #     " --train.run_name=fullbatch-mse0-shuffled-adam-lr3e-3",
]
# =======================================================================

LOG_DIR = f"logs/{JOB_NAME}"            # relative to WORK_DIR


def slug(text: str) -> str:
    """Filesystem-safe tag for log filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:80]


def resolve_run_names():
    """Best-effort: resolve each entry's train.run_name via the config package
    (like submit_sco_vanilla.py) to catch output-dir collisions BEFORE
    submitting. Falls back to raw args if the local env can't import config
    (e.g. broken numpy on the dev box) -- the job env is what actually matters."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import config as cfgmod
        names = []
        for args in RUNS:
            toks = args.split()
            cfg = cfgmod.apply_overrides(cfgmod.load(toks[0]), toks[1:])
            names.append(cfg.train.run_name)
        return names, True
    except Exception as exc:  # noqa: BLE001 - purely a pre-submit convenience
        print(f"[warn] could not resolve run_names locally ({exc}); "
              f"skipping collision check")
        return [slug(a) for a in RUNS], False


def build_launcher(run_names) -> str:
    """The bash script the job runs: launch every run pinned to its own GPU,
    tail -f all logs into the job stdout so stream-logs stays useful, then wait
    on each pid and fail the job if any run failed.

    This is a real multi-line script (not an inlined --command one-liner)
    because `A && B & C &` parses as `(A && B) & (C) &` -- the cd/export get
    captured by the first background subshell and every later run starts in the
    container's default cwd with the wrong PATH."""
    lines = [
        "#!/usr/bin/env bash",
        "# generated by submit_sco_fullbatch.py -- records exactly what this job ran",
        "set -u",
        f"cd {WORK_DIR} || exit 1",
        f'export PATH="{CONDA_ENV_PATH}/bin:$PATH"',
        f"mkdir -p {LOG_DIR} || exit 1",
        'echo "[fullbatch] host=$(hostname) cwd=$(pwd)"',
        f'{ENV_PYTHON} -c "import torch; print(\'[fullbatch] torch\', torch.__version__, '
        f"'cuda_available', torch.cuda.is_available(), 'device_count', torch.cuda.device_count())\"",
        "",
    ]
    logs = []
    for i, (args, name) in enumerate(zip(RUNS, run_names)):
        log = f"{LOG_DIR}/gpu{i}-{slug(name)}.log"
        logs.append(log)
        lines += [
            f"# --- GPU {i}: {args}",
            f"CUDA_VISIBLE_DEVICES={i} {ENV_PYTHON} -u train_fullbatch.py {args} "
            f"> {log} 2>&1 &",
            f"PID{i}=$!",
            f'echo "[fullbatch] GPU {i} pid $PID{i}: {args}"',
        ]
    lines += [
        "",
        "sleep 5",
        f"tail -n +1 -f {' '.join(logs)} &",
        "TAILPID=$!",
        "",
        "rc=0",
    ]
    for i, args in enumerate(RUNS):
        lines += [
            f"if wait $PID{i}; then",
            f'  echo "[fullbatch] GPU {i} OK: {args}"',
            "else",
            f'  echo "[fullbatch] GPU {i} FAILED: {args}"; rc=1',
            "fi",
        ]
    lines += [
        "",
        "sleep 2",
        "kill $TAILPID 2>/dev/null",
        'echo "[fullbatch] all runs finished, rc=$rc"',
        "exit $rc",
        "",
    ]
    return "\n".join(lines)


def write_launcher(run_names) -> str:
    """Write the launcher next to the logs and return its absolute path."""
    log_dir_abs = os.path.join(WORK_DIR, LOG_DIR)
    os.makedirs(log_dir_abs, exist_ok=True)
    path = os.path.join(log_dir_abs, "launch.sh")
    with open(path, "w") as f:
        f.write(build_launcher(run_names))
    os.chmod(path, 0o755)
    return path


def submit_job(command: str) -> bool:
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
        "--priority", PRIORITY,
        "--quota-type", QUOTA_TYPE,
        "--command", command,
    ]
    print(f"Submitting job: {JOB_NAME}")
    print(f"Command: {command}")
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

    if not RUNS:
        sys.exit("RUNS is empty -- nothing to submit.")
    if len(RUNS) > N_GPUS:
        sys.exit(f"RUNS has {len(RUNS)} entries but the node has {N_GPUS} GPUs; "
                 f"split into multiple jobs.")

    run_names, resolved = resolve_run_names()
    if resolved and len(set(run_names)) != len(run_names):
        sys.exit(f"run_name collision: {run_names}\n"
                 f"give sweep entries a unique --train.run_name.")

    print(f"Job name:   {JOB_NAME}")
    print(f"Profile:    {PROFILE}")
    print(f"Worker spec:{WORKER_SPEC}")
    print(f"Work dir:   {WORK_DIR}")
    print(f"Python:     {ENV_PYTHON}")
    print(f"Layout:     {len(RUNS)} single-process full-batch run(s), "
          f"one per GPU (no torchrun)")
    for i, (a, n) in enumerate(zip(RUNS, run_names)):
        print(f"  GPU {i}: {a}   -> runs/{n}/")

    launcher = write_launcher(run_names)
    print(f"Launcher:   {launcher}")
    if not args.yes:
        if input("Continue? (y/n): ").strip().lower() != "y":
            print("Cancelled.")
            return
    if not submit_job(f"bash {launcher}"):
        sys.exit(1)


if __name__ == "__main__":
    main()
