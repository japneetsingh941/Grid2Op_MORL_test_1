#!/bin/bash
#SBATCH --job-name=ppo_orch
#SBATCH --partition=cpu
#SBATCH --account=cai_res
#SBATCH --time=0-12:00:00
# Training uses NUM_CORE = min(cpu_count(), 16) environments + the main process,
# so ~17 CPUs are actually used. The old request of 80 CPUs / 128G meant only two
# jobs fit per node (192 logical CPUs), which is why array tasks sat in PD with
# reason (Resources) -- it was never a job-count policy cap.
# Re-check --mem against `sacct -j <id> --format=MaxRSS` and set to ~1.5x actual.
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --array=0-9
#SBATCH --chdir=/cluster/home/singhjap/Projects/Grid2Op_MORL_test_1
#SBATCH --output=/cluster/home/singhjap/Projects/Grid2Op_MORL_test_1/logs/%x_%A_%a.out
#SBATCH --error=/cluster/home/singhjap/Projects/Grid2Op_MORL_test_1/logs/%x_%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail
set -x

# ---------- Paths ----------
# ---------- Paths ----------
BASE_DIR="/cluster/home/singhjap/Projects/Grid2Op_MORL_test_1"

PROJECT_DIR="$BASE_DIR"
LOG_DIR="$BASE_DIR/logs"
VENV_DIR="$BASE_DIR/.venvs/morl_env_py311"
PIP_CACHE="$BASE_DIR/.pip-cache"

mkdir -p "$LOG_DIR" "$PIP_CACHE"

# ---------- Threads ----------
# Deliberately NOT SLURM_CPUS_PER_TASK: the 16 env workers are separate
# processes, so giving each one that many BLAS threads oversubscribes the node
# and makes concurrent runs fight each other.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- Array identity ----------
echo "[SLURM] SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-N/A}"
echo "[SLURM] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-N/A}"

# Concurrent array tasks share one working directory, so they must not all wipe
# and repopulate submission/ppo-ckpt and then evaluate whatever won the race.
export ORCH_SKIP_STAGES="deploy_checkpoint,run_runner"

# ---------- Modules ----------
module purge
module load python/3.11

# ---------- Venv ----------
if [ -d "$VENV_DIR" ]; then
  echo "[INFO] Reusing venv at $VENV_DIR"
else
  echo "[SETUP] Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ---------- Keep installs inside venv only ----------
unset PYTHONPATH
export PYTHONNOUSERSITE=1
unset PYTHONUSERBASE PIP_TARGET PIP_PREFIX
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install -U pip wheel setuptools --cache-dir "$PIP_CACHE" --no-input

# ---------- Core stack (Torch first; no requirements.txt pass) ----------
python - <<'PY'
import subprocess, sys
def pip(*args): subprocess.check_call([sys.executable,'-m','pip',*args])

# 1) Torch + CUDA 12.4 (bundled, no CUDA module needed)
pip('install','--index-url','https://download.pytorch.org/whl/cu124',
    'torch==2.5.1+cu124','torchvision==0.20.1+cu124','torchaudio==2.5.1+cu124')

# 2) Fast backend & core grid stack
pip('install',
    'numpy>=1.26,<2.2',
    'scipy>=1.14,<1.17',
    'lightsim2grid==0.10.3',
    'grid2op==1.12.1',
    'tensorflow>=2.16,<2.20',
    'numba')   # optional speedup (used by PandaPower backend too)

# 3) MORL & friends
pip('install',
    'gymnasium==0.29.1',
    'mo-gymnasium==1.1.0',
    'morl_baselines==1.0.0',
    'cvxpy==1.6.4',
    'scikit-learn==1.6.1',
    'h5py==3.13.0',
    'wandb')
PY

# ---------- Print where things actually are ----------
python - <<'PY'
import sys, torch, grid2op, lightsim2grid, numpy, tensorflow
print("[PYTHON]", sys.executable)
print("[NUMPY ]", numpy.__version__)
print("[TORCH ]", torch.__version__, "cuda?", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())
if torch.cuda.is_available(): print("[GPU   ]", torch.cuda.get_device_name(0))
print("[G2OP  ]", grid2op.__version__)
print("[LS2G  ]", lightsim2grid.__version__)
PY

# ---------- Warm up env (detect np.NaN regression; fallback to numpy<2 if needed) ----------
set +e
python - <<'PY'
import sys
from lightsim2grid import LightSimBackend
import grid2op

try:
    env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
    print("[GRID2OP] Loaded:", env.name, "lines:", env.n_line)
    env.close()
except AttributeError as e:
    msg = str(e)
    # NumPy 2.0 removed np.NaN; older LS2G used it -> detect and signal fallback
    if "np.NaN was removed" in msg or "np.NaN" in msg:
        print("NP_NAN_BREAKAGE_DETECTED")
        sys.exit(42)
    raise
PY
RC=$?
set -e

if [ "$RC" -eq 42 ]; then
  echo "[FALLBACK] Detected np.NaN problem; switching to numpy<2 for this job."
  python -m pip install 'numpy==1.26.4' --no-input --cache-dir "$PIP_CACHE"
  # Re-warm
  python - <<'PY'
from lightsim2grid import LightSimBackend
import grid2op
env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
print("[GRID2OP] Re-Loaded:", env.name, "lines:", env.n_line)
env.close()
PY
fi

# ---------- Weights & Biases (assumes you've run `wandb login` once) ----------
export WANDB_API_KEY="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"
export WANDB_PROJECT="grid2op"
export WANDB_ENTITY="japneet-singh-can-university-of-toronto"
export WANDB_MODE=online
export WANDB_DIR="$LOG_DIR/wandb"
# One group per array submission: all 10 repeats land in the same group, so W&B
# can plot mean +/- band across seeds via "Group by -> group".
export WANDB_RUN_GROUP="sweep_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
export WANDB_NAME="ols_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID:-0}"
export WANDB_TAGS="grid2op,PPO,NOMORL"

# ---------- Run ----------
cd "$PROJECT_DIR"
echo "[RUN] Start $(date)"
python -u orchestrate_training.py

echo "[RUN] End $(date)"
