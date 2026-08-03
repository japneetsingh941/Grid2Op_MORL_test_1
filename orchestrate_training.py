import os
import json
import subprocess
import sys
import shutil
from pathlib import Path
import math
import time

import sweep_config

# Global safety limit for the whole pipeline (23h30)
MAX_RUNTIME_SECONDS = ((12)-0.5) * 3600

def log(msg: str, verbose: bool = True):
    if verbose:
        print(f"[ORCH] {msg}")


def merge_csvs(base_dir: Path, pattern: str, final_name: str, verbose: bool = True):
    """
    Concatenate all CSV files matching `pattern` in `base_dir` into `final_name`.
    Teachers write with header=False, so we can just byte-concatenate.
    """
    files = sorted(base_dir.glob(pattern))
    if not files:
        log(f"[WARN] No CSV files found for pattern {pattern} in {base_dir}", verbose)
        return

    final_path = base_dir / final_name
    log(f"Merging {len(files)} files into {final_path}", verbose)

    with final_path.open("wb") as fout:
        for f in files:
            with f.open("rb") as fin:
                shutil.copyfileobj(fin, fout)

def delete_matching_files(base_dir: Path, pattern: str, verbose: bool = True):
    """
    Delete all files in `base_dir` that match the given glob `pattern`.
    """
    removed = 0
    for f in base_dir.glob(pattern):
        try:
            f.unlink()
            removed += 1
            log(f"Deleted file: {f}", verbose)
        except Exception as e:
            log(f"[WARN] Could not delete {f}: {e}", verbose)
    if removed == 0:
        log(f"No files to delete for pattern {pattern} in {base_dir}", verbose)



def log(msg: str, verbose: bool = True):
    if verbose:
        print(f"[ORCH] {msg}")


def run_step(python_exe, script_path: Path, cwd: Path, extra_args=None, verbose=True):
    extra_args = extra_args or []
    cmd = [python_exe, str(script_path)] + extra_args
    log(f"Running: {' '.join(cmd)} (cwd={cwd})", verbose)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def build_senior_run_specs(cfg, python_exe, script_path: Path, log_dir: Path,
                           extra_args=None, array_task_id=None, job_id=None):
    """
    One spec per training run to launch inside this SLURM job.

    The cluster caps the number of concurrent *jobs* (MaxJobs), not CPUs, so
    several runs share a single job instead of taking a job slot each. Each run
    gets:
      - its own identity (RUN_TAG in sweep mode, else a globally unique
        RUN_INDEX), which the training script turns into its ./log*, ./ckpt*
        and W&B run name;
      - its own log file, so the runs don't interleave into one SLURM log;
      - an `srun` step pinned to ONE node: a run's envs are local
        SingleEnvMultiProcess forks, so a run must never straddle nodes.

    Two modes:
      - sweep (`sweep.enabled`): the array task id IS the permutation index, and
        this job holds that permutation's `runs_per_permutation` runs, all
        sharing one alpha ordering and one W&B group.
      - plain: `parallel.runs_per_job` runs indexed from the array task id.

    Returns a list of dicts: {run_index, tag, group, cmd, env_overrides, log_path}.
    Pure function (no side effects) so the layout and srun flags are testable.
    """
    extra_args = list(extra_args or [])
    parallel_cfg = cfg.get("parallel", {})

    cpus_per_run = int(parallel_cfg.get("cpus_per_run", 0) or 0)
    mem_per_cpu = parallel_cfg.get("mem_per_cpu")
    in_slurm = job_id not in (None, "")
    use_srun = bool(parallel_cfg.get("use_srun_steps", False)) and in_slurm
    # --exact needs Slurm >= 20.11; override via config if srun rejects a flag.
    srun_flags = list(parallel_cfg.get("srun_flags")
                      or ["--nodes=1", "--ntasks=1", "--exact"])

    if sweep_config.enabled(cfg):
        runs = sweep_config.sweep_runs(cfg, array_task_id)
    else:
        runs_per_job = max(1, int(parallel_cfg.get("runs_per_job", 1) or 1))
        try:
            task_offset = int(array_task_id) * runs_per_job
        except (TypeError, ValueError):
            task_offset = 0
        runs = [{"index": task_offset + i} for i in range(runs_per_job)]

    specs = []
    for run in runs:
        run_index = run["index"]
        tag = run.get("tag")

        if use_srun:
            cmd = ["srun"] + srun_flags
            if cpus_per_run > 0:
                cmd.append(f"--cpus-per-task={cpus_per_run}")
            if mem_per_cpu:
                cmd.append(f"--mem-per-cpu={mem_per_cpu}")
            cmd += [python_exe, str(script_path)] + extra_args
        else:
            cmd = [python_exe, str(script_path)] + extra_args

        env_overrides = {"RUN_INDEX": str(run_index)}
        if tag:
            # Sweep run: tag drives log/ckpt dirs and the W&B run name, the
            # order drives the curriculum, the group keeps the arm together.
            env_overrides["RUN_TAG"] = tag
            env_overrides["RUN_GROUP"] = run["group"]
            env_overrides["RUN_ALPHA_ORDER"] = ",".join(run["order"])
        if in_slurm and cpus_per_run > 0:
            # Each run must see its own share, not the whole job allocation:
            # SLURM_CPUS_PER_TASK is what resolve_num_envs() clamps envs against.
            # Only set inside a real job — never fake SLURM vars off-cluster.
            env_overrides["SLURM_CPUS_PER_TASK"] = str(cpus_per_run)

        specs.append({
            "run_index": run_index,
            "tag": tag,
            "group": run.get("group"),
            "cmd": cmd,
            "env_overrides": env_overrides,
            "log_path": log_dir / f"senior_{tag or f'run{run_index}'}.out",
        })

    return specs


def launch_senior_runs(specs, cwd: Path, verbose=True):
    """
    Start every spec concurrently and wait for all of them.

    Same launch/wait pattern as the Teacher split stages above: collect the
    processes, wait on each, and fail loudly if any exited non-zero.
    """
    procs = []
    for spec in specs:
        env = os.environ.copy()
        for key, value in spec["env_overrides"].items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value

        spec["log_path"].parent.mkdir(parents=True, exist_ok=True)
        logf = spec["log_path"].open("w")
        label = spec.get("tag") or f"run{spec['run_index']}"
        log(f"Launching senior run {label}: {' '.join(spec['cmd'])} "
            f"(log={spec['log_path']})", verbose)
        p = subprocess.Popen(
            spec["cmd"], cwd=str(cwd), env=env,
            stdout=logf, stderr=subprocess.STDOUT,
        )
        procs.append((spec, p, logf))

    failures = []
    for spec, p, logf in procs:
        ret = p.wait()
        logf.close()
        label = spec.get("tag") or f"run{spec['run_index']}"
        if ret != 0:
            failures.append((label, ret, spec["log_path"]))
            log(f"[WARN] Senior run {label} exited with code {ret} "
                f"(see {spec['log_path']})", verbose)
        else:
            log(f"Senior run {label} finished OK", verbose)

    if failures:
        details = ", ".join(f"{label}=code{ret}" for label, ret, _ in failures)
        raise RuntimeError(f"{len(failures)}/{len(procs)} senior runs failed: {details}")


def copy_tree(src: Path, dst: Path, wipe: bool, verbose=True):
    if not src.exists():
        log(f"[WARN] Source checkpoint directory does not exist: {src}", verbose)
        return

    if wipe and dst.exists():
        log(f"Removing existing target dir: {dst}", verbose)
        shutil.rmtree(dst)

    log(f"Copying checkpoint from {src} to {dst}", verbose)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    # --- resolve paths ---
    repo_root = Path(__file__).resolve().parent
    # --- start global runtime clock ---
    start_time = time.time()
    # Make start time visible to all child processes (Teachers, Junior, Senior, etc.)
    os.environ.setdefault("ORCH_START_TIME", str(start_time))

    def check_time_and_maybe_stop(stage_name: str) -> bool:
        """
        Returns False if the global max runtime is exceeded.
        In that case we skip this and all remaining stages.
        """
        elapsed = time.time() - start_time
        if elapsed > MAX_RUNTIME_SECONDS:
            print(f"[ORCH] Reached {MAX_RUNTIME_SECONDS/3600:.2f}h time budget before '{stage_name}'.", flush=True)
            print("[ORCH] Skipping this and all subsequent stages; exiting orchestrator.", flush=True)
            return False
        return True


    # --- load config ---
    cfg_path = repo_root / "config_orchestrator.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    verbose = cfg.get("general", {}).get("verbose", True)
    python_exe = cfg.get("general", {}).get("python_executable") or sys.executable

    paths = cfg.get("paths", {})
    stages = cfg.get("stages", {})

    teacher_dir = repo_root / paths.get("teacher_dir", "Teacher")
    tutor_dir = repo_root / paths.get("tutor_dir", "Tutor")
    junior_dir = repo_root / paths.get("junior_dir", "JuniorStudent")
    senior_dir = repo_root / paths.get("senior_dir", "SeniorStudent")
    submission_dir = repo_root / paths.get("submission_dir", "submission")

    # --- Stage 1: Teacher module (parallel splits) ---
    if stages.get("run_teacher1", False):
        if not check_time_and_maybe_stop("Teacher1"):
            return
        print(f"Stage 1: Teacher1 module (parallel splits)", flush=True)
        script1 = teacher_dir / "Teacher1.py"

        # Clean up any old Teacher1 job outputs before starting
        delete_matching_files(teacher_dir, "Experiences1_job*.csv", verbose=verbose)

        N_SPLITS_T1 = 13
        TOTAL_EPISODES_T1 = 1000
        EPISODES_PER_SPLIT = math.ceil(TOTAL_EPISODES_T1 / N_SPLITS_T1)

        teacher_procs = []
        base_env = os.environ.copy()
        # Limit per-process threading so 10 processes don't fight too much
        base_env.setdefault("OMP_NUM_THREADS", "6")
        base_env.setdefault("MKL_NUM_THREADS", "6")

        for split_id in range(N_SPLITS_T1):
            start = split_id * EPISODES_PER_SPLIT
            end = min(start + EPISODES_PER_SPLIT, TOTAL_EPISODES_T1)
            if start >= TOTAL_EPISODES_T1:
                break
            job_id = f"job{split_id}"

            cmd = [
                python_exe,
                str(script1),
                "--episode-start", str(start),
                "--episode-end", str(end),
                "--job-id", job_id,
            ]
            log(f"Launching Teacher1 split {split_id}: {cmd}", verbose)
            p = subprocess.Popen(cmd, cwd=str(teacher_dir), env=base_env)
            teacher_procs.append(p)

        # Wait for all Teacher1 splits
        for p in teacher_procs:
            ret = p.wait()
            if ret != 0:
                raise RuntimeError(f"Teacher1 split exited with code {ret}")

        # Merge per-job CSVs into a single Experiences1.csv
        merge_csvs(teacher_dir, pattern="Experiences1_job*.csv", final_name="Experiences1.csv", verbose=verbose)
        # Optional: delete per-job CSVs after successful merge
        delete_matching_files(teacher_dir, "Experiences1_job*.csv", verbose=verbose)

    if stages.get("run_teacher2", False):
        if not check_time_and_maybe_stop("Teacher2"):
            return
        print(f"Stage 1: Teacher2 module (parallel splits)", flush=True)
        script2 = teacher_dir / "Teacher2.py"
        # Clean up any old Teacher2 job outputs before starting
        delete_matching_files(teacher_dir, "Experiences2_job*.csv", verbose=verbose)

        N_SPLITS_T2 = 13
        TOTAL_EPISODES_T2 = 100
        EPISODES_PER_SPLIT = math.ceil(TOTAL_EPISODES_T2 / N_SPLITS_T2)

        teacher_procs = []
        base_env = os.environ.copy()
        base_env.setdefault("OMP_NUM_THREADS", "6")
        base_env.setdefault("MKL_NUM_THREADS", "6")

        for split_id in range(N_SPLITS_T2):
            start = split_id * EPISODES_PER_SPLIT
            end = min(start + EPISODES_PER_SPLIT, TOTAL_EPISODES_T2)
            if start >= TOTAL_EPISODES_T2:
                break
            job_id = f"job{split_id}"

            cmd = [
                python_exe,
                str(script2),
                "--episode-start", str(start),
                "--episode-end", str(end),
                "--job-id", job_id,
            ]
            log(f"Launching Teacher2 split {split_id}: {cmd}", verbose)
            p = subprocess.Popen(cmd, cwd=str(teacher_dir), env=base_env)
            teacher_procs.append(p)

        for p in teacher_procs:
            ret = p.wait()
            if ret != 0:
                raise RuntimeError(f"Teacher2 split exited with code {ret}")

        merge_csvs(teacher_dir, pattern="Experiences2_job*.csv", final_name="Experiences2.csv", verbose=verbose)
        # Optional: delete per-job CSVs after successful merge
        delete_matching_files(teacher_dir, "Experiences2_job*.csv", verbose=verbose)

    if stages.get("generate_action_space", False):
        if not check_time_and_maybe_stop("Generate_action_space"):
            return
        print(f"Stage 1: Generate_action_space module", flush=True)
        script = teacher_dir / "Generate_action_space.py"

        # In original README they call this from repo root, but we can also
        # set cwd to ActionSpace; adjust to whichever your file expects.
        run_step(python_exe, script, script.parent, verbose=verbose)

    # --- Stage 2: Tutor module ---
    if stages.get("generate_tutor_dataset", False):
        if not check_time_and_maybe_stop("Tutor_generate_dataset"):
            return
        print(f"Stage 2: Tutor module", flush=True)
        script = tutor_dir / "Generate_teaching_dataset.py"
        run_step(python_exe, script, tutor_dir, verbose=verbose)

    # --- Stage 3: JuniorStudent module ---
    jr_cfg = cfg.get("junior_student", {})
    train_args = jr_cfg.get("train_args", ["Train"])
    convert_args = jr_cfg.get("convert_args", ["Convert"])

    junior_script = junior_dir / "JuniorStudent.py"

    if stages.get("junior_train", False):
        if not check_time_and_maybe_stop("JuniorStudent_train"):
            return
        print(f"Stage 3: junior_train", flush=True)
        run_step(python_exe, junior_script, junior_dir, extra_args=train_args, verbose=verbose)

    if stages.get("junior_convert", False):
        if not check_time_and_maybe_stop("JuniorStudent_convert"):
            return
        print(f"Stage 3: junior_convert", flush=True)
        run_step(python_exe, junior_script, junior_dir, extra_args=convert_args, verbose=verbose)

    # --- Stage 4: SeniorStudent (PPO) ---
    sr_cfg = cfg.get("senior_student", {})
    sr_args = sr_cfg.get("script_args", [])

    if stages.get("senior_train", False):
        if not check_time_and_maybe_stop("SeniorStudent_train"):
            return
        if stages.get("pereference_condition", False):
            print(f"Stage 4: senior_train_pereference_condition_morl", flush=True)
            script = senior_dir / "SeniorStudentPrefConMORL.py"
            run_step(python_exe, script, senior_dir, extra_args=sr_args, verbose=verbose)
        elif stages.get("gated_tiered_morl", False):
            print(f"Stage 4: senior_train_gated_tiered_morl", flush=True)
            script = senior_dir / "SeniorStudentMORL.py"

            runs_per_job = max(1, int(cfg.get("parallel", {}).get("runs_per_job", 1) or 1))
            if sweep_config.enabled(cfg) or runs_per_job > 1:
                # Pack several runs into this one job: the cluster limits the
                # number of concurrent jobs, not CPUs. In sweep mode the array
                # task id selects which alpha permutation this job trains.
                specs = build_senior_run_specs(
                    cfg,
                    python_exe,
                    script,
                    log_dir=repo_root / "logs",
                    extra_args=sr_args,
                    array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                    job_id=os.environ.get("SLURM_JOB_ID"),
                )
                if specs[0].get("tag"):
                    order = specs[0]["env_overrides"]["RUN_ALPHA_ORDER"]
                    print(f"Stage 4: launching {len(specs)} senior runs in this job "
                          f"| alpha order: {order} | wandb group: {specs[0]['group']}",
                          flush=True)
                else:
                    print(f"Stage 4: launching {len(specs)} senior runs in this job "
                          f"(indices {specs[0]['run_index']}-{specs[-1]['run_index']})",
                          flush=True)
                launch_senior_runs(specs, senior_dir, verbose=verbose)
            else:
                run_step(python_exe, script, senior_dir, extra_args=sr_args, verbose=verbose)
        elif stages.get("do_nothing", False):
            print(f"Stage 4: senior_do_nothing", flush=True)
            script = senior_dir / "SeniorStudent_DoNothingBaseline.py"
            run_step(python_exe, script, senior_dir, extra_args=sr_args, verbose=verbose)
        elif stages.get("random_action", False):
            print(f"Stage 4: senior_random_action", flush=True)
            script = senior_dir / "SeniorStudent_RandomAction.py"
            run_step(python_exe, script, senior_dir, extra_args=sr_args, verbose=verbose)
        else:
            print(f"Stage 4: senior_train", flush=True)
            script = senior_dir / "SeniorStudent.py"
            run_step(python_exe, script, senior_dir, extra_args=sr_args, verbose=verbose)

    # --- Stage 5: Deploy chosen checkpoint into submission/ppo-ckpt ---
    if stages.get("deploy_checkpoint", False):
        if not check_time_and_maybe_stop("deploy_checkpoint"):
            return
        print(f"Stage 5: deploy_checkpoint", flush=True)
        dep_cfg = cfg.get("deployment", {})

        src_rel = dep_cfg.get("ckpt_source_dir", "SeniorStudent/ckpt/ppo_best")
        dst_rel = dep_cfg.get("ckpt_target_dir", "submission/ppo-ckpt")
        wipe_target = dep_cfg.get("wipe_target_before_copy", True)

        src = repo_root / src_rel
        dst = repo_root / dst_rel

        copy_tree(src, dst, wipe=wipe_target, verbose=verbose)

    # --- Optional: run runner.py to evaluate ---
    if stages.get("run_runner", False):
        if not check_time_and_maybe_stop("run_runner"):
            return
        print(f"Optional Stage: run_runner", flush=True)
        script = repo_root / "runner.py"
        run_step(python_exe, script, repo_root, verbose=verbose)

    log("Pipeline finished.", verbose)


if __name__ == "__main__":
    main()
