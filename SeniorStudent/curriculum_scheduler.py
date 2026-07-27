import copy
import wandb

_last_stage = None


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

    # Find the current stage
    current_stage = stages[0]
    stage_index = 0

    for i, stage in enumerate(stages):
        if global_step >= stage["step"]:
            current_stage = stage
            stage_index = i
        else:
            break

    # Apply scheduled weights
    for key, value in current_stage.get("weights", {}).items():
        weights[key] = value

    # Only log when the stage changes
    if stage_index != _last_stage:

        print(
            f"[Scheduler] Stage {stage_index} at step {global_step}",
            flush=True
        )

        print(
            f"[Scheduler] Active weights: {current_stage['weights']}",
            flush=True
        )

        # Log to W&B
        if wandb.run is not None:
            wandb.log(
                {
                    "curriculum/stage": stage_index,
                    "curriculum/alpha_struct": weights.get("alpha_struct"),
                    "curriculum/alpha_fair": weights.get("alpha_fair"),
                    "curriculum/alpha_sust": weights.get("alpha_sust"),
                },
                step=global_step,
            )

        _last_stage = stage_index

    return weights