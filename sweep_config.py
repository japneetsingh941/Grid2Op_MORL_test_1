"""
Alpha-ordering sweep: every permutation of the MORL objectives, repeated N times.

The permutation fixes the order the objectives are INTRODUCED in; the alpha weight
is then shared equally among everything introduced so far, so an ordering
[A, B, C] trains 1/0/0 -> 0.5/0.5/0 -> 1/3 each. What varies between permutations
is which objective gets the undivided weight first, and which one only ever
appears in the final equal split.

Each permutation gets its own W&B group (e.g. alpha_distribute_sus_fair_struc) and
its runs are spread over one or more SLURM array tasks: an array task holds
`parallel.runs_per_job` runs, so the job's CPU footprint can be made small enough
to schedule without changing what any single run gets.

Single source of truth for the sweep layout: imported by orchestrate_training.py
(to build the run specs), by submit.sh (to size --array / --ntasks) and by
SeniorStudent/run_config.py (to synthesise the curriculum for one run).
"""
import itertools
import json
from pathlib import Path

# Objective -> weight key in the `morl` / curriculum weight dicts
OBJECTIVES = ["struct", "fair", "sust"]
WEIGHT_KEY = {obj: f"alpha_{obj}" for obj in OBJECTIVES}
# Short forms used in group names and run tags
ABBREV = {"struct": "struc", "fair": "fair", "sust": "sus"}

DEFAULT_GROUP_PREFIX = "alpha_distribute"
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
    return list(_sweep_cfg(cfg).get("objectives") or OBJECTIVES)


def abbrev(cfg: dict) -> dict:
    merged = dict(ABBREV)
    merged.update(_sweep_cfg(cfg).get("abbrev") or {})
    return merged


def stage_steps(cfg: dict) -> list:
    return list(_sweep_cfg(cfg).get("stage_steps") or DEFAULT_STAGE_STEPS)


def sweep_orders(cfg: dict) -> list:
    """
    All orderings of the objectives (6 for three objectives), or the explicit
    `sweep.orders` subset if the config provides one.
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


def curriculum_from_order(order, steps=None) -> dict:
    """
    Weight-sharing curriculum for one ordering: each stage introduces one more
    objective and splits the alpha weight equally over everything introduced so
    far, so objectives are added rather than swapped out.

    For an ordering [A, B, C] that is

        stage 0:  A=1.0     B=0.0     C=0.0
        stage 1:  A=0.5     B=0.5     C=0.0
        stage 2:  A=1/3     B=1/3     C=1/3

    The final stage is identical for every permutation; the orderings differ in
    which objective is learned alone first and how long each one has been active
    by the time training ends.

    Shaped like the JSON `curriculum` block so
    curriculum_scheduler.get_scheduled_weights() consumes it unchanged.
    """
    order = list(order)
    steps = list(steps or DEFAULT_STAGE_STEPS)
    if len(steps) < len(order):
        raise ValueError(
            f"Need at least {len(order)} stage steps for order {order}, got {steps}"
        )

    stages = []
    for i in range(len(order)):
        # Objectives order[0..i] are active, each holding an equal share.
        share = 1.0 / (i + 1)
        weights = {WEIGHT_KEY[obj]: 0.0 for obj in order}
        for obj in order[: i + 1]:
            weights[WEIGHT_KEY[obj]] = share
        stages.append({"step": int(steps[i]), "weights": weights})

    return {"enabled": True, "stages": stages}


def order_from_env(value: str) -> list:
    """Parse RUN_ALPHA_ORDER ("struct,fair,sust") into a list."""
    return [part.strip() for part in str(value).split(",") if part.strip()]
