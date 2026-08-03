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
CPUS_PER_RUN="$(read_parallel_cfg cpus_per_run 20)"
MEM_PER_RUN="$(read_parallel_cfg mem_per_run 32G)"
ARRAY_MAX=$((NUM_PARALLEL_RUNS - 1))

echo "[SUBMIT] parallel runs : $NUM_PARALLEL_RUNS (array 0-$ARRAY_MAX)"
echo "[SUBMIT] cpus per run  : $CPUS_PER_RUN"
echo "[SUBMIT] mem per run   : $MEM_PER_RUN"
echo "[SUBMIT] wandb group   : $(read_wandb_group)"

sbatch \
    --export=ALL,BASE_DIR="$DIR" \
    --chdir="$DIR" \
    --array=0-"$ARRAY_MAX" \
    --cpus-per-task="$CPUS_PER_RUN" \
    --mem="$MEM_PER_RUN" \
    --output="$DIR/logs/%x_%A_%a.out" \
    --error="$DIR/logs/%x_%A_%a.err" \
    run_orchestrator_jap.SLURM
