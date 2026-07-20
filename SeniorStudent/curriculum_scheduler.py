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

    # Get the configuration name for this stage
    config_name = current_stage["config"]

    # Retrieve all stored MORL configurations
    configs = curriculum_cfg.get("morl_configs", {})

    if config_name not in configs:
        raise KeyError(
            f"Unknown curriculum configuration '{config_name}'"
        )

    # Load the selected configuration
    stage_weights = configs[config_name]

    # Override the default weights
    for key, value in stage_weights.items():

        if key not in weights:
            raise KeyError(
                f"Unknown MORL weight '{key}' "
                f"in curriculum configuration '{config_name}'"
            )

        weights[key] = value

    # Print only when the stage changes
    if stage_index != _last_stage:

        print(
            f"[Scheduler] Stage {stage_index} "
            f"({config_name}) at step {global_step}",
            flush=True
        )

        print(
            f"[Scheduler] Active weights:",
            flush=True
        )

        for key, value in stage_weights.items():
            print(f"    {key}: {value}", flush=True)

        _last_stage = stage_index

    return weights