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

NUM_PARALLEL_RUNS="$(read_parallel_cfg num_parallel_runs 10)"
RUNS_PER_JOB="$(read_parallel_cfg runs_per_job 10)"
CPUS_PER_RUN="$(read_parallel_cfg cpus_per_run 20)"
MEM_PER_CPU="$(read_parallel_cfg mem_per_cpu 1500M)"

# The association caps concurrent JOBS (MaxJobs=5), not CPUs, so pack
# RUNS_PER_JOB training runs into each job and submit as few jobs as possible.
NUM_JOBS=$(( (NUM_PARALLEL_RUNS + RUNS_PER_JOB - 1) / RUNS_PER_JOB ))
ARRAY_MAX=$(( NUM_JOBS - 1 ))
CPUS_PER_JOB=$(( RUNS_PER_JOB * CPUS_PER_RUN ))

# SUBMIT_ARRAY=1-1 ./submit.sh -> shift the array indices. RUN_INDEX is
# array_task_id * runs_per_job + i, so this keeps a new submission's log/ckpt
# dirs clear of runs already on disk (task 1 x 10 runs -> ckpt_run10..19).
ARRAY_SPEC="${SUBMIT_ARRAY:-0-$ARRAY_MAX}"

echo "[SUBMIT] total runs        : $NUM_PARALLEL_RUNS"
echo "[SUBMIT] jobs              : $NUM_JOBS (array $ARRAY_SPEC)"
echo "[SUBMIT] runs per job      : $RUNS_PER_JOB"
echo "[SUBMIT] cpus per run      : $CPUS_PER_RUN"
echo "[SUBMIT] total cpus per job: $CPUS_PER_JOB"
echo "[SUBMIT] mem per cpu       : $MEM_PER_CPU"
echo "[SUBMIT] wandb group       : $(read_wandb_group)"

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
