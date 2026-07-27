#!/bin/bash

DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$DIR/logs"

sbatch \
    --export=ALL,BASE_DIR="$DIR" \
    --chdir="$DIR" \
    --output="$DIR/logs/%x_%j.out" \
    --error="$DIR/logs/%x_%j.err" \
    run_orchestrator_jap.SLURM