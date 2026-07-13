_last_stage = None


def get_stage(global_step):
    global _last_stage

    if global_step < 10:
        stage = 0
    elif global_step < 20:
        stage = 1
    else:
        stage = 2

    if stage != _last_stage:
        print(
            f"\n==============================\n"
            f"Curriculum Stage {stage}\n"
            f"Global Step: {global_step}\n"
            f"==============================\n"
        )
        _last_stage = stage

    return stage