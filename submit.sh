#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="$DIR/config_orchestrator.json"

mkdir -p "$DIR/logs"

# Read a value from the "parallel" section of config_orchestrator.json,
# falling back to the given default when it is missing/empty.
read_parallel_cfg() {
    python3 - "$CFG" "$1" "$2" <<'PY'
import json, sys
cfg_path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(cfg_path) as f:
        value = json.load(f).get("parallel", {}).get(key)
except Exception:
    value = None
print(default if value in (None, "") else value)
PY
}

read_wandb_group() {
    python3 - "$CFG" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f:
        print(json.load(f).get("wandb", {}).get("group") or "<slurm array job id>")
except Exception:
    print("<unknown>")
PY
}

# Sweep mode: sweep_config.py is the single source of truth for the array size,
# the runs per job and the per-permutation W&B group names.
read_sweep() {
    python3 - "$1" <<'PY'
import sys
sys.path.insert(0, ".")
import sweep_config
cfg = sweep_config.load_cfg()
what = sys.argv[1]
if what == "enabled":
    print("1" if sweep_config.enabled(cfg) else "0")
elif what == "permutations":
    print(sweep_config.num_permutations(cfg))
elif what == "runs_per_permutation":
    print(sweep_config.runs_per_permutation(cfg))
elif what == "array_tasks":
    print(sweep_config.num_array_tasks(cfg))
elif what == "runs_per_job":
    print(sweep_config.runs_per_job(cfg))
elif what == "chunks":
    print(sweep_config.chunks_per_permutation(cfg))
elif what == "total_runs":
    print(sweep_config.total_runs(cfg))
elif what == "groups":
    for order in sweep_config.sweep_orders(cfg):
        print(sweep_config.group_name(cfg, order))
PY
}

NUM_PARALLEL_RUNS="$(read_parallel_cfg num_parallel_runs 10)"
RUNS_PER_JOB="$(read_parallel_cfg runs_per_job 10)"
CPUS_PER_RUN="$(read_parallel_cfg cpus_per_run 20)"
MEM_PER_CPU="$(read_parallel_cfg mem_per_cpu 1500M)"

SWEEP_ENABLED="$(read_sweep enabled)"

if [ "$SWEEP_ENABLED" = "1" ]; then
    # Each permutation's runs are split into chunks of runs_per_job, one array
    # task per chunk. Job size (runs_per_job x cpus_per_run) is what has to fit
    # the partition's idle CPUs; runs per permutation stays whatever the
    # experiment needs.
    NUM_JOBS="$(read_sweep array_tasks)"
    RUNS_PER_JOB="$(read_sweep runs_per_job)"
    TOTAL_RUNS="$(read_sweep total_runs)"
    CHUNKS="$(read_sweep chunks)"
    RUNS_PER_PERM="$(read_sweep runs_per_permutation)"
else
    # The association caps concurrent JOBS (MaxJobs=5), not CPUs, so pack
    # RUNS_PER_JOB training runs into each job and submit as few jobs as possible.
    NUM_JOBS=$(( (NUM_PARALLEL_RUNS + RUNS_PER_JOB - 1) / RUNS_PER_JOB ))
    TOTAL_RUNS="$NUM_PARALLEL_RUNS"
fi

ARRAY_MAX=$(( NUM_JOBS - 1 ))
CPUS_PER_JOB=$(( RUNS_PER_JOB * CPUS_PER_RUN ))

# SUBMIT_ARRAY=3-3 ./submit.sh -> submit a subset of the array. In sweep mode the
# array index IS the permutation index, so this re-runs a single permutation; in
# plain mode it shifts RUN_INDEX (array_task_id * runs_per_job + i) to keep new
# log/ckpt dirs clear of runs already on disk.
ARRAY_SPEC="${SUBMIT_ARRAY:-0-$ARRAY_MAX}"

echo "[SUBMIT] total runs        : $TOTAL_RUNS"
echo "[SUBMIT] jobs              : $NUM_JOBS (array $ARRAY_SPEC)"
echo "[SUBMIT] runs per job      : $RUNS_PER_JOB"
echo "[SUBMIT] cpus per run      : $CPUS_PER_RUN"
echo "[SUBMIT] total cpus per job: $CPUS_PER_JOB"
echo "[SUBMIT] mem per cpu       : $MEM_PER_CPU"
if [ "$SWEEP_ENABLED" = "1" ]; then
    echo "[SUBMIT] sweep             : config-ordering permutations"
    echo "[SUBMIT] runs per perm     : $RUNS_PER_PERM (split into $CHUNKS chunk(s) of $RUNS_PER_JOB)"
    read_sweep groups | nl -v0 -w1 -s': ' | sed 's/^/[SUBMIT]   perm /'
else
    echo "[SUBMIT] wandb group       : $(read_wandb_group)"
fi

# SUBMIT_TEST_ONLY=1 ./submit.sh  -> validate the allocation without queueing.
# Plain string (not an array): expanding an empty array trips `set -u` on bash 3.x.
TEST_ONLY=""
if [ "${SUBMIT_TEST_ONLY:-0}" != "0" ]; then
    echo "[SUBMIT] --test-only (nothing will be queued)"
    TEST_ONLY="--test-only"
fi

# shellcheck disable=SC2086  # intentional: empty TEST_ONLY must expand to nothing
sbatch $TEST_ONLY \
    --export=ALL,BASE_DIR="$DIR" \
    --chdir="$DIR" \
    --array="$ARRAY_SPEC" \
    --ntasks="$RUNS_PER_JOB" \
    --cpus-per-task="$CPUS_PER_RUN" \
    --mem-per-cpu="$MEM_PER_CPU" \
    --output="$DIR/logs/%x_%A_%a.out" \
    --error="$DIR/logs/%x_%A_%a.err" \
    run_orchestrator_jap.SLURM
