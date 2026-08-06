"""
Config-ordering sweep: every permutation of the named MORL configs, repeated N times.

A "config" is a complete set of MORL weights - all ten w_* metric weights,
tau_primary and the three alpha_* block weights - defined once under
`sweep.configs` in config_orchestrator.json. The sweep permutes the ORDER in which
those configs are applied as curriculum stages, so each permutation trains the same
three configs in a different sequence.

Each permutation gets its own W&B group (e.g. config_sur_fair_sust) and its runs
are spread over one or more SLURM array tasks: an array task holds
`parallel.runs_per_job` runs, so the job's CPU footprint can be made small enough
to schedule without changing what any single run gets.

Single source of truth for the sweep layout: imported by orchestrate_training.py
(to build the run specs), by submit.sh (to size --array / --ntasks) and by
SeniorStudent/run_config.py (to synthesise the curriculum for one run).
"""
import copy
import itertools
import json
from pathlib import Path

# Config names, each resolved through `sweep.configs` to a full weight dict
CONFIGS = ["survival", "fairness", "sustainability"]
# Short forms used in group names and run tags
ABBREV = {"survival": "sur", "fairness": "fair", "sustainability": "sust"}

DEFAULT_GROUP_PREFIX = "config"
DEFAULT_RUNS_PER_PERMUTATION = 10
DEFAULT_STAGE_STEPS = [0, 400000, 800000]


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def load_cfg() -> dict:
    cfg_path = repo_root() / "config_orchestrator.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sweep_cfg(cfg: dict) -> dict:
    return cfg.get("sweep", {}) or {}


def enabled(cfg: dict) -> bool:
    return bool(_sweep_cfg(cfg).get("enabled", False))


def objectives(cfg: dict) -> list:
    """Config names taking part in the sweep, in their canonical order."""
    return list(_sweep_cfg(cfg).get("objectives") or CONFIGS)


def configs(cfg: dict) -> dict:
    """
    name -> full MORL weight dict, from `sweep.configs`.

    Every name used in `sweep.orders` must be defined here: a stage overlays only
    the keys it carries, so a partial config would silently leave the previous
    stage's weights in place.
    """
    return dict(_sweep_cfg(cfg).get("configs") or {})


def abbrev(cfg: dict) -> dict:
    merged = dict(ABBREV)
    merged.update(_sweep_cfg(cfg).get("abbrev") or {})
    return merged


def stage_steps(cfg: dict) -> list:
    return list(_sweep_cfg(cfg).get("stage_steps") or DEFAULT_STAGE_STEPS)


def sweep_orders(cfg: dict) -> list:
    """
    All orderings of the configs (6 for three configs), or the explicit
    `sweep.orders` list if the config provides one.

    The explicit form also fixes the order the array tasks are laid out in, which
    is how the baseline ordering is pushed to the highest array indices.
    """
    explicit = _sweep_cfg(cfg).get("orders")
    if explicit:
        return [list(order) for order in explicit]
    return [list(order) for order in itertools.permutations(objectives(cfg))]


def num_permutations(cfg: dict) -> int:
    return len(sweep_orders(cfg))


def runs_per_permutation(cfg: dict) -> int:
    value = _sweep_cfg(cfg).get("runs_per_permutation", DEFAULT_RUNS_PER_PERMUTATION)
    return max(1, int(value or DEFAULT_RUNS_PER_PERMUTATION))


def total_runs(cfg: dict) -> int:
    return num_permutations(cfg) * runs_per_permutation(cfg)


def runs_per_job(cfg: dict) -> int:
    """
    How many runs one SLURM job holds, i.e. `parallel.runs_per_job` capped at
    runs_per_permutation.

    Decoupled from runs_per_permutation on purpose: the job's CPU footprint is
    runs_per_job * cpus_per_run, and a partition with only N idle CPUs cannot
    start a job bigger than that no matter how many job slots are free. Smaller
    jobs fit in gaps; the experiment each run sees is unchanged.
    """
    value = cfg.get("parallel", {}).get("runs_per_job") or runs_per_permutation(cfg)
    return max(1, min(int(value), runs_per_permutation(cfg)))


def chunks_per_permutation(cfg: dict) -> int:
    per_job = runs_per_job(cfg)
    return (runs_per_permutation(cfg) + per_job - 1) // per_job


def num_array_tasks(cfg: dict) -> int:
    return num_permutations(cfg) * chunks_per_permutation(cfg)


def order_slug(cfg: dict, order) -> str:
    short = abbrev(cfg)
    return "_".join(short.get(obj, obj) for obj in order)


def group_name(cfg: dict, order) -> str:
    prefix = _sweep_cfg(cfg).get("group_prefix") or DEFAULT_GROUP_PREFIX
    return f"{prefix}_{order_slug(cfg, order)}"


def run_tag(cfg: dict, order, repeat: int) -> str:
    """Per-run identity: drives ./log*, ./ckpt* and the W&B run name."""
    return f"{order_slug(cfg, order)}_r{repeat}"


def sweep_runs(cfg: dict, task_index) -> list:
    """
    The runs belonging to ONE array task.

    An array task is (permutation, chunk): each permutation's runs_per_permutation
    runs are split into chunks of runs_per_job, so several smaller jobs cover one
    permutation. The group is per permutation regardless of chunking, and `tag`
    uses the GLOBAL repeat number, so r0..r9 stay unique across chunks.

    Returns [{order, group, tag, repeat, index}, ...] of length <= runs_per_job
    (the last chunk is short when the split is uneven).
    """
    orders = sweep_orders(cfg)
    chunks = chunks_per_permutation(cfg)
    per_job = runs_per_job(cfg)
    per_perm = runs_per_permutation(cfg)
    tasks = len(orders) * chunks

    try:
        index = int(task_index)
    except (TypeError, ValueError):
        index = 0

    if not 0 <= index < tasks:
        raise ValueError(
            f"Array task index {index} out of range: the sweep defines "
            f"{len(orders)} permutations x {chunks} chunk(s) = {tasks} array "
            f"tasks (valid indices 0-{tasks - 1})"
        )

    perm_index, chunk_index = divmod(index, chunks)
    order = orders[perm_index]
    group = group_name(cfg, order)

    first = chunk_index * per_job
    last = min(first + per_job, per_perm)

    return [
        {
            "order": list(order),
            "group": group,
            "tag": run_tag(cfg, order, repeat),
            "repeat": repeat,
            "index": perm_index * per_perm + repeat,
        }
        for repeat in range(first, last)
    ]


def curriculum_from_order(cfg: dict, order, steps=None) -> dict:
    """
    Curriculum for one ordering: stage i applies the config named order[i] from
    step steps[i] onwards.

    Each stage carries the COMPLETE weight dict, so nothing leaks through from the
    base `morl` block or from the previous stage - get_scheduled_weights() overlays
    exactly the keys a stage contains.

    Shaped like the JSON `curriculum` block so
    curriculum_scheduler.get_scheduled_weights() consumes it unchanged.
    """
    order = list(order)
    steps = list(steps or DEFAULT_STAGE_STEPS)
    if len(steps) < len(order):
        raise ValueError(
            f"Need at least {len(order)} stage steps for order {order}, got {steps}"
        )

    defined = configs(cfg)
    missing = [name for name in order if name not in defined]
    if missing:
        raise KeyError(
            f"Order {order} references config(s) {missing} that are not defined in "
            f"sweep.configs (defined: {sorted(defined)})"
        )

    stages = [
        {
            "name": name,
            "step": int(steps[i]),
            "weights": copy.deepcopy(defined[name]),
        }
        for i, name in enumerate(order)
    ]

    return {"enabled": True, "stages": stages}


def order_from_env(value: str) -> list:
    """Parse RUN_ALPHA_ORDER ("survival,fairness,sustainability") into a list."""
    return [part.strip() for part in str(value).split(",") if part.strip()]
