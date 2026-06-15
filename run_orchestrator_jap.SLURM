#!/bin/bash
#SBATCH --job-name=ppo_orch
#SBATCH --partition=cpu
#SBATCH --account=cai_res
#SBATCH --time=0-12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=80
#SBATCH --chdir=/cluster/home/singhjap/Projects/Grid2Op_MORL_test_1
#SBATCH --output=/cluster/home/singhjap/Projects/Grid2Op_MORL_test_1/logs/%x_%j.out
#SBATCH --error=/cluster/home/singhjap/Projects/Grid2Op_MORL_test_1/logs/%x_%j.err
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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
export WANDB_NAME="ols_${SLURM_JOB_ID}"
export WANDB_TAGS="grid2op,PPO,NOMORL"

# ---------- Run ----------
cd "$PROJECT_DIR"
echo "[RUN] Start $(date)"
python -u orchestrate_training.py

echo "[RUN] End $(date)"
