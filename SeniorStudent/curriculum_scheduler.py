import copy

_last_stage = None

def get_scheduled_weights(global_step, base_weights):
    global _last_stage

    weights = copy.deepcopy(base_weights)

    if global_step < 10:
        stage = 0
        weights["alpha_struct"] = 1.0

    elif global_step < 20:
        stage = 1
        weights["alpha_struct"] = 0.5

    else:
        stage = 2
        weights["alpha_struct"] = 0.0

    if stage != _last_stage:
        print("\n========================")
        print(f"Stage {stage}")
        print(f"Step {global_step}")
        print(f"alpha_struct = {weights['alpha_struct']}")
        print("========================\n")
        _last_stage = stage

    return weights