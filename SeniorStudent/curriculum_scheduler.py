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

        stage_name = current_stage.get("name", "")

        print(
            f"[Scheduler] Stage {stage_index}"
            f"{f' ({stage_name})' if stage_name else ''} at step {global_step}",
            flush=True
        )

        print(
            f"[Scheduler] Active weights: {current_stage['weights']}",
            flush=True
        )

        # Log to W&B. A stage is a whole config (alphas + tau + every metric
        # weight), so log all of it - otherwise the charts show the alphas
        # switching while the metric weights that changed with them are invisible.
        if wandb.run is not None:
            payload = {"curriculum/stage": stage_index}
            if stage_name:
                payload["curriculum/stage_name"] = stage_name
            for key in ("alpha_struct", "alpha_fair", "alpha_sust", "tau_primary",
                        "w_fair_rho", "w_fair_curt", "w_equity", "w_ren", "w_co2",
                        "w_risk", "w_n1", "w_econ", "w_simplicity", "w_l2rpn"):
                value = weights.get(key)
                if value is not None:
                    payload[f"curriculum/{key}"] = value
            wandb.log(payload, step=global_step)

        _last_stage = stage_index

    return weights