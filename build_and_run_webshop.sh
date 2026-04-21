#!/bin/bash
# ============================================================
# WebShop EnvTuning Docker usage helper
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-webshop-envtuning:latest}"
IMAGE_TAR="${IMAGE_TAR:-webshop-envtuning.tar.gz}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/path/to/Qwen3-8B}"
MODEL_CONTAINER_PATH="${MODEL_CONTAINER_PATH:-/models/Qwen3-8B}"
OUTPUT_HOST_PATH="${OUTPUT_HOST_PATH:-/path/to/outputs}"

build_image() {
    docker build \
        -f "$ROOT/Dockerfile.webshop" \
        -t "$IMAGE" \
        "$ROOT"
}

export_image() {
    docker save "$IMAGE" | gzip > "$IMAGE_TAR"
    echo "Image saved to $IMAGE_TAR"
    echo "Size: $(du -sh "$IMAGE_TAR" | awk '{print $1}')"
}

load_image() {
    docker load < "$IMAGE_TAR"
}

run_script() {
    local script="$1"
    local env_args=(
        -e MODEL_PATH="$MODEL_CONTAINER_PATH"
        -e CKPT_DIR=/workspace/outputs/checkpoints
        -e OUTPUT_ROOT=/workspace/outputs
        -e LOG_DIR=/workspace/outputs/logs
        -e N_GPUS="${N_GPUS:-8}"
        -e TP="${TP:-2}"
        -e WEBSHOP_USE_SMALL="${WEBSHOP_USE_SMALL:-true}"
    )
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        env_args+=(-e WANDB_API_KEY="$WANDB_API_KEY")
    fi

    mkdir -p "$OUTPUT_HOST_PATH"
    docker run --gpus all --ipc=host --shm-size=64g \
        --ulimit nproc=65536:65536 \
        --ulimit nofile=1048576:1048576 \
        "${env_args[@]}" \
        -v "$MODEL_HOST_PATH:$MODEL_CONTAINER_PATH:ro" \
        -v "$OUTPUT_HOST_PATH:/workspace/outputs" \
        "$IMAGE" \
        bash "$script"
}

run_vanilla() {
    run_script scripts/run_webshop_vanilla.sh
}

run_obs_aug() {
    run_script scripts/run_webshop_envtuning.sh
}

run_full_envtuning() {
    run_script scripts/run_webshop_full_envtuning.sh
}

usage() {
    echo "Usage:"
    echo "  IMAGE=webshop-envtuning:latest bash build_and_run_webshop.sh build"
    echo "  bash build_and_run_webshop.sh export"
    echo "  bash build_and_run_webshop.sh load"
    echo ""
    echo "Run experiments after setting MODEL_HOST_PATH and OUTPUT_HOST_PATH:"
    echo "  MODEL_HOST_PATH=/path/to/Qwen3-8B OUTPUT_HOST_PATH=/path/to/outputs bash build_and_run_webshop.sh vanilla"
    echo "  MODEL_HOST_PATH=/path/to/Qwen3-8B OUTPUT_HOST_PATH=/path/to/outputs bash build_and_run_webshop.sh obs-aug"
    echo "  MODEL_HOST_PATH=/path/to/Qwen3-8B OUTPUT_HOST_PATH=/path/to/outputs bash build_and_run_webshop.sh full-envtuning"
    echo ""
    echo "Optional env vars: N_GPUS, TP, WANDB_API_KEY"
}

case "${1:-}" in
    build) build_image ;;
    export) export_image ;;
    load) load_image ;;
    vanilla) run_vanilla ;;
    obs-aug) run_obs_aug ;;
    full-envtuning) run_full_envtuning ;;
    *) usage; exit 2 ;;
esac
