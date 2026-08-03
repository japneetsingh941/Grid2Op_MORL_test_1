"""
Helpers that read run-level settings (parallelism + W&B naming) from
config_orchestrator.json so they can be changed without touching the code.

Used by the SeniorStudent MORL training scripts.
"""
import importlib.util
import json
import os
from multiprocessing import cpu_count
from pathlib import Path

DEFAULT_PROJECT = "vt1_grid2op_senior_ppo"
DEFAULT_RUN_NAME_PREFIX = "senior_student"
# Fallback cap when nothing is configured (keeps the OS "open files" limit happy)
DEFAULT_MAX_ENVS = 16


def repo_root() -> Path:
    """Repo root = parent of the SeniorStudent directory holding this file."""
    return Path(__file__).resolve().parent.parent


def config_path() -> Path:
    return repo_root() / "config_orchestrator.json"


def load_orchestrator_cfg() -> dict:
    cfg_path = config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def allocated_cpus() -> int:
    """
    CPUs this process may actually use.

    Inside a SLURM job array, cpu_count() reports the whole node, not the
    cgroup slice, so 10 concurrent tasks would each spawn envs for all cores.
    Prefer SLURM_CPUS_PER_TASK, then the CPU affinity mask.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # macOS / Windows
        return max(1, cpu_count())


def load_sweep_config():
    """
    Load sweep_config.py from the repo root (same importlib pattern the training
    script uses for morl_objectives.py).
    """
    path = repo_root() / "sweep_config.py"
    spec = importlib.util.spec_from_file_location("sweep_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_curriculum(cfg: dict, verbose: bool = True) -> dict:
    """
    Curriculum for this run.

    In sweep mode the orchestrator passes RUN_ALPHA_ORDER (e.g. "sust,fair,struct")
    and the stages are synthesised from it, so all 6 permutations share one code
    path and the JSON `curriculum` block stays untouched. Without that env var the
    configured curriculum is used, exactly as before.
    """
    order_env = os.environ.get("RUN_ALPHA_ORDER")
    if not order_env:
        return cfg.get("curriculum", {})

    sweep = load_sweep_config()
    order = sweep.order_from_env(order_env)
    curriculum = sweep.curriculum_from_order(order, sweep.stage_steps(cfg))
    if verbose:
        print(f"[RUN_CFG] alpha order: {' -> '.join(order)}", flush=True)
        for i, stage in enumerate(curriculum["stages"]):
            print(f"[RUN_CFG]   stage {i} @ step {stage['step']}: {stage['weights']}",
                  flush=True)
    return curriculum


def resolve_run_index() -> str:
    """
    Identity of this training run, used for ./log*, ./ckpt* and the W&B run name.

    RUN_TAG (sweep mode, e.g. "sus_fair_struc_r3") wins so dirs are readable and
    can never collide across permutations. RUN_INDEX is set by the orchestrator
    when several runs are packed into one SLURM job (they all share one array
    task id and would otherwise collide). Falls back to the array task id, then
    to "" for a plain standalone run.
    """
    run_tag = os.environ.get("RUN_TAG")
    if run_tag not in (None, ""):
        return str(run_tag)

    run_index = os.environ.get("RUN_INDEX")
    if run_index not in (None, ""):
        return str(run_index)

    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_task_id not in (None, ""):
        return str(array_task_id)

    return ""


def resolve_num_envs(cfg: dict, verbose: bool = True) -> int:
    """
    Number of parallel grid2op environments for this run, from
    `parallel.num_envs` in the config, clamped to the CPUs we were given.
    """
    parallel_cfg = cfg.get("parallel", {})
    cpus = allocated_cpus()

    requested = parallel_cfg.get("num_envs")
    if requested is None:
        requested = min(cpus, DEFAULT_MAX_ENVS)

    num_envs = max(1, min(int(requested), cpus))
    if verbose:
        print(
            f"[RUN_CFG] num_envs={num_envs} "
            f"(requested={requested}, allocated_cpus={cpus})",
            flush=True,
        )
    return num_envs


def apply_thread_limits(cfg: dict, verbose: bool = True) -> int:
    """
    Cap BLAS/OpenMP threads per environment process. With many env processes per
    task, the default (threads = all cores) makes them fight for the same CPUs.
    """
    parallel_cfg = cfg.get("parallel", {})
    threads = int(parallel_cfg.get("threads_per_env", 1) or 1)
    threads = max(1, threads)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
        os.environ[var] = str(threads)
    if verbose:
        print(f"[RUN_CFG] threads_per_env={threads}", flush=True)
    return threads


def resolve_wandb_cfg(cfg: dict, run_suffix: str = "", timestamp: str = "") -> dict:
    """
    Resolve W&B project / group / run name / tags.

    Precedence: RUN_GROUP (sweep) -> config_orchestrator.json -> environment ->
    built-in default. An empty or null `wandb.group` falls back to the SLURM
    array job id, so a plain array submission still groups its tasks together.
    """
    w = cfg.get("wandb", {})

    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    run_index = os.environ.get("RUN_INDEX")

    project = w.get("project") or os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT
    # RUN_GROUP is set per permutation in sweep mode and must win over the
    # single group configured in JSON.
    group = (os.environ.get("RUN_GROUP") or w.get("group")
             or os.environ.get("WANDB_GROUP") or array_job_id or "local")
    prefix = w.get("run_name_prefix") or DEFAULT_RUN_NAME_PREFIX

    name = w.get("run_name")
    if not name:
        parts = [prefix]
        if run_suffix:
            parts.append(run_suffix.lstrip("_"))
        if timestamp:
            parts.append(timestamp)
        name = "_".join(parts)

    tags = [str(t) for t in (w.get("tags") or [])]
    if array_task_id is not None:
        tags.append(f"task{array_task_id}")
    if run_index not in (None, ""):
        tags.append(f"run{run_index}")

    return {
        "enabled": bool(w.get("enabled", True)),
        "project": project,
        "group": group,
        "name": name,
        "tags": tags,
        "notes": w.get("notes") or None,
        "entity": w.get("entity") or os.environ.get("WANDB_ENTITY") or None,
        "job_type": w.get("job_type") or "senior_train",
    }
