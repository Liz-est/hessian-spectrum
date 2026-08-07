#!/usr/bin/env python3
"""
Train + analyze the four pos0 frozen-layer 0-layer MSE presets on the
zipf-imbalanced synth data (random-init isn't in yet; init is still all-ones):

  1. mse0-pos0-frozen_embd-sgd-lr0p1-imb-init1G, mse0-pos0-frozen_embd-adamw-lr1p5e-3-imb-init1G
  2. mse0-pos0-frozen_lmhead-sgd-lr0p05-imb-init1G, mse0-pos0-frozen_lmhead-adamw-lr1p5e-3-imb-init1G

Two batches, at most two of our jobs at a time, sequential.

Lessons baked in (see 2026-07-27 postmortem):
  * Job names are derived from the preset key, so they are unique and match
    what actually runs. Free-form names once got crossed (an SGD preset ran
    under an "-adamw" job name) and duplicated within a batch.
  * Jobs are polled by their pt- job ID, never by name. `sco acp jobs list`
    keeps terminal jobs from earlier submissions around; a name->state dict
    let a stale same-name FAILED row shadow the fresh run (and a stale
    SUCCEEDED row end the batch wait after one poll, breaking the
    two-at-a-time queue).
  * purge resolves the preset key to analyze.files_name / train.run_name via
    config.load() -- the output dirs are NOT the preset key in general, and
    purging the key can be a silent no-op that lets analyze_vanilla.py's
    resume logic skip stale results.

    nohup python3 -u sco_run_mse0.py > sco_mse0.log 2>&1 &
"""

import os
import re
import shutil
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
WORKER_SPEC = "n6ls.iu.i40.8.32c512g"   # 8x H100
STORAGE_MOUNT = "01995892-d478-76d8-aec7-13fd8284477e:/data"

USER_DATA = "/data/250010020"
REPO_ROOT = f"{USER_DATA}/hessian-spectrum"
WORK_DIR = f"{REPO_ROOT}/toy_models"
CONDA_ENV_PATH = f"{USER_DATA}/miniconda3/envs/nanogpt"
ENV_PYTHON = f"{CONDA_ENV_PATH}/bin/python"
NPROC_PER_NODE = 8

# preset keys only; the job name is derived from the key. Batches of at most
# two, sequential.
BATCHES = [
    # # ---- SGD fine-tune (3 pts): 0.007, 0.013, 0.017 ----
    # ["REP1-frz_lmhead-sgd-lr0p007-G02",
    #  "REP1-frz_lmhead-sgd-lr0p013-G02"],
    # ["REP1-frz_lmhead-sgd-lr0p017-G02",
    #  "REP1-frz_lmhead-adam-lr1e-6-G02"],
    # # ---- Adam geometric sweep (6 pts): 1e-6..5e-5 ----
    # ["REP1-frz_lmhead-adam-lr2e-6-G02",
    #  "REP1-frz_lmhead-adam-lr5e-6-G02"],
    # ["REP1-frz_lmhead-adam-lr1e-5-G02",
    #  "REP1-frz_lmhead-adam-lr2e-5-G02"],
    ["REP-mserep-pos0-frz_embd-fullbs-sgd-lr0p05-imb-initG02-nobias",
     "REP-mserep-pos0-frz_embd-fullbs-sgd-lr0p07-imb-initG02-nobias"],
    # ---- Muon geometric sweep (5 pts): 2e-5..4e-4 ----
    ["REP-mserep-pos0-frz_embd-fullbs-sgd-lr0p09-imb-initG02-nobias",
     "REP-mserep-pos0-frz_embd-fullbs-adam-lr6e-6-imb-initG02-nobias"],
    ["REP-mserep-pos0-frz_embd-fullbs-adam-lr9e-6-imb-initG02-nobias",
      "REP-mserep-pos0-frz_embd-fullbs-adam-lr2e-6-imb-initG02-nobias"],

    
]

POLL_SECONDS = 300
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "KILLED"}
JOB_ID_RE = re.compile(r"\bpt-[a-z0-9]+\b")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def job_name_for(preset):
    """Unique-per-submission job name derived from the preset key (so the
    platform list is self-describing even across reruns of the same preset)."""
    stamp = time.strftime("%m%d%H%M")
    # SCO job names: keep it to alnum plus - and _ just in case.
    base = re.sub(r"[^A-Za-z0-9_-]", "-", preset)
    return f"{base}-{stamp}"


def resolve_dirs(preset):
    """Preset key -> (analyze.files_name, train.run_name): the ACTUAL output
    dir names, which are not the preset key in general."""
    import config
    c = config.load(preset)
    return c.analyze.files_name, c.train.run_name


def purge_all_outputs(preset):
    """Full wipe of files/<files_name> (fresh run: every layer changes)."""
    files_name, run_name = resolve_dirs(preset)
    d = os.path.join(WORK_DIR, "files", files_name)
    if os.path.isdir(d):
        shutil.rmtree(d)
        log(f"removed files/{files_name}/ entirely")
    run_dir = os.path.join(WORK_DIR, "runs", run_name)
    for stale in ("val_from_ckpts.csv", "loss_curve_with_val.png",
                  "val_by_freq.csv", "val_by_freq.png",
                  "rep_groups.csv", "rep_groups.png"):
        p = os.path.join(run_dir, stale)
        if os.path.exists(p):
            os.remove(p)
            log(f"removed stale runs/{run_name}/{stale}")


def command_for(preset):
    analyze = (f"{ENV_PYTHON} -u -m torch.distributed.run --standalone "
               f"--nproc_per_node={NPROC_PER_NODE} analyze_vanilla.py {preset}")
    train = (f"{ENV_PYTHON} -u -m torch.distributed.run --standalone "
             f"--nproc_per_node={NPROC_PER_NODE} train_vanilla_transformer.py {preset}")
    return (f"cd {WORK_DIR} && "
            f"export PATH={CONDA_ENV_PATH}/bin:$PATH && "
            f"{train} && {analyze}")


def list_jobs():
    """All rows from `sco acp jobs list`, newest first: [(id, name, state)]."""
    r = subprocess.run(
        [SCO, "--profile", PROFILE, "acp", "jobs", "list",
         "--workspace-name", WORKSPACE_NAME],
        capture_output=True, text=True)
    rows = []
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        # data rows: ID | NAME | USER | RESOURCE | NODE | STATE | ...
        if len(parts) >= 6 and parts[0].startswith("pt-"):
            rows.append((parts[0], parts[1], parts[5]))
    return rows


def submit(preset):
    """Submit one job; returns (job_id, job_name) or (None, job_name)."""
    job_name = job_name_for(preset)
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
        "--command", command_for(preset),
    ]
    log(f"submitting {job_name} ({preset})")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"SUBMIT FAILED {job_name}: {r.stdout} {r.stderr}")
        return None, job_name
    # prefer the job ID echoed by `jobs create`; fall back to the newest
    # list row with our (unique) name.
    m = JOB_ID_RE.search(r.stdout or "")
    job_id = m.group(0) if m else None
    if job_id is None:
        time.sleep(5)
        for jid, name, _state in list_jobs():   # newest first
            if name == job_name:
                job_id = jid
                break
    if job_id is None:
        log(f"WARNING: submitted {job_name} but could not resolve its job ID")
        return None, job_name
    log(f"submitted {job_name} as {job_id}")
    return job_id, job_name


def wait_batch(jobs):
    """jobs: {job_id: job_name}. Poll BY ID until every job is terminal."""
    while True:
        time.sleep(POLL_SECONDS)
        states = {jid: state for jid, _name, state in list_jobs()}
        cur = {jobs[jid]: states.get(jid, "UNKNOWN") for jid in jobs}
        log(f"poll: {cur}")
        if all(s in TERMINAL for s in cur.values()):
            return cur


def main():
    results = {}
    for i, batch in enumerate(BATCHES, 1):
        log(f"===== batch {i}/{len(BATCHES)}: {batch} =====")
        jobs = {}   # job_id -> job_name
        for preset in batch:
            purge_all_outputs(preset)
            job_id, job_name = submit(preset)
            if job_id is None:
                results[job_name] = "SUBMIT_FAILED"
            else:
                jobs[job_id] = job_name
        if not jobs:
            log(f"batch {i}: nothing submitted, moving on")
            continue
        final = wait_batch(jobs)
        results.update(final)
        log(f"batch {i} finished: {final}")

    log("===== all batches done =====")
    for name, state in results.items():
        log(f"  {name}: {state}")
    if any(s != "SUCCEEDED" for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
