"""
Alpha-ordering sweep: every permutation of the MORL objectives, repeated N times.

Each permutation is one SLURM array task holding `runs_per_permutation` packed
runs, and gets its own W&B group (e.g. base_vector_sus_fair_struc).

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

DEFAULT_GROUP_PREFIX = "base_vector"
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


def order_slug(cfg: dict, order) -> str:
    short = abbrev(cfg)
    return "_".join(short.get(obj, obj) for obj in order)


def group_name(cfg: dict, order) -> str:
    prefix = _sweep_cfg(cfg).get("group_prefix") or DEFAULT_GROUP_PREFIX
    return f"{prefix}_{order_slug(cfg, order)}"


def run_tag(cfg: dict, order, repeat: int) -> str:
    """Per-run identity: drives ./log*, ./ckpt* and the W&B run name."""
    return f"{order_slug(cfg, order)}_r{repeat}"


def sweep_runs(cfg: dict, permutation_index) -> list:
    """
    The runs belonging to ONE array task (= one permutation).

    Returns [{order, group, tag, repeat, index}, ...] of length
    runs_per_permutation. `index` is the run's global position across the sweep,
    kept only for logging/traceability - isolation uses `tag`.
    """
    orders = sweep_orders(cfg)

    try:
        perm_index = int(permutation_index)
    except (TypeError, ValueError):
        perm_index = 0

    if not 0 <= perm_index < len(orders):
        raise ValueError(
            f"Permutation index {perm_index} out of range: the sweep defines "
            f"{len(orders)} permutations (valid array indices 0-{len(orders) - 1})"
        )

    order = orders[perm_index]
    per_perm = runs_per_permutation(cfg)
    group = group_name(cfg, order)

    return [
        {
            "order": list(order),
            "group": group,
            "tag": run_tag(cfg, order, repeat),
            "repeat": repeat,
            "index": perm_index * per_perm + repeat,
        }
        for repeat in range(per_perm)
    ]


def curriculum_from_order(order, steps=None) -> dict:
    """
    Pure-sequential curriculum for one ordering: at each stage boundary exactly
    one objective is active at weight 1.0 and the others are 0.0.

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
    for i, active in enumerate(order):
        weights = {WEIGHT_KEY[obj]: 0.0 for obj in order}
        weights[WEIGHT_KEY[active]] = 1.0
        stages.append({"step": int(steps[i]), "weights": weights})

    return {"enabled": True, "stages": stages}


def order_from_env(value: str) -> list:
    """Parse RUN_ALPHA_ORDER ("struct,fair,sust") into a list."""
    return [part.strip() for part in str(value).split(",") if part.strip()]
