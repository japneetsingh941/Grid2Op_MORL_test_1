import copy

try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False

_last_stage = None

# Cadence for curriculum logging, in env steps. The curriculum is a step
# function, so it has to be sampled densely -- logging only on stage change
# gives W&B three points per run, which it renders as straight diagonals
# between the plateaus instead of a staircase.
_LOG_EVERY = 1000


def _stage_label(stage, index):
    """Human-readable name for a stage; falls back to the index."""
    return stage.get("name") or f"stage_{index}"


def get_scheduled_weights(global_step, base_weights, curriculum_cfg):
    """
    Returns a copy of the MORL weights updated according to the
    curriculum configuration.
    """

    global _last_stage

    # Start from the default MORL weights
    weights = copy.deepcopy(base_weights)

    # Curriculum disabled -> use original weights
    if not curriculum_cfg.get("enabled", False):
        return weights

    stages = curriculum_cfg.get("stages", [])
    if _last_stage is None:
        print(f"[DEBUG] Curriculum stages: {stages}", flush=True)

    if len(stages) == 0:
        return weights

    # The scan below breaks at the first future stage, so an out-of-order
    # config would silently select the wrong stage.
    stages = sorted(stages, key=lambda s: s["step"])

    # Find the current stage
    current_stage = stages[0]
    stage_index = 0

    for i, stage in enumerate(stages):
        if global_step >= stage["step"]:
            current_stage = stage
            stage_index = i
        else:
            break

    # Apply every scheduled weight

    stage_weights = current_stage.get("weights", {})

    for key, value in stage_weights.items():

        if key not in weights:
            raise KeyError(
                f"Unknown MORL weight '{key}' "
                f"in curriculum stage {stage_index}"
            )

        weights[key] = value

    changed = stage_index != _last_stage

    # Print only when the stage changes
    if changed:
        print(
            f"[Scheduler] Stage {stage_index} "
            f"({_stage_label(current_stage, stage_index)}) at step {global_step}",
            flush=True
        )
        print(
            f"[Scheduler] Active weights: {current_stage['weights']}",
            flush=True
        )
        _last_stage = stage_index

    # Log on the transition itself plus at a fixed cadence, so the charts
    # render as plateaus with sharp edges rather than interpolated ramps.
    if USE_WANDB and wandb.run is not None:
        if changed or global_step % _LOG_EVERY == 0:
            payload = {
                "curriculum/stage": stage_index,
                "curriculum/stage_name": _stage_label(current_stage, stage_index),
                # Spikes to 1 only on the step where the stage flips.
                "curriculum/stage_changed": 1 if changed else 0,
                "curriculum/alpha_struct": weights.get("alpha_struct"),
                "curriculum/alpha_fair": weights.get("alpha_fair"),
                "curriculum/alpha_sust": weights.get("alpha_sust"),
            }

            # One-hot per stage, emitted for *every* stage so each regime has a
            # complete 0/1 line from step 0. This is the "which configuration
            # is running" chart.
            for i, stage in enumerate(stages):
                label = _stage_label(stage, i)
                payload[f"curriculum/regime/{label}"] = 1 if i == stage_index else 0

            wandb.log(payload, step=global_step)

    return weights
