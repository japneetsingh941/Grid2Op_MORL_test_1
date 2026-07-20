import copy

_last_stage = None


def get_scheduled_weights(global_step, base_weights, curriculum_cfg):
    """
    Returns a copy of the MORL weights updated according to the
    curriculum configuration.

    The scheduler starts from the default MORL weights and overrides
    them using the named configuration specified for the current stage.
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

        # Support both the old and new JSON formats

    if "weights" in current_stage:
        # Old format
        stage_weights = current_stage["weights"]
        config_name = current_stage.get("name", f"Stage {stage_index}")

    elif "config" in current_stage:
        # New format
        config_name = current_stage["config"]

        configs = curriculum_cfg.get("morl_configs", {})

        if config_name not in configs:
            raise KeyError(
                f"Unknown curriculum configuration '{config_name}'"
            )

        stage_weights = configs[config_name]

    else:
        raise KeyError(
            "Curriculum stage must contain either "
            "'weights' or 'config'."
        )

    return weights