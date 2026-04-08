#!/bin/bash
# ============================================================
# ALFWorld EnvTuning Docker 使用指南
# ============================================================

# === 第一步：在当前机器上构建镜像 ===
# 大约需要 10-15 分钟
build_image() {
    cd /home/nvidia/env2scaffold
    docker build -t alfworld-envtuning:latest .
}

# === 第二步：导出镜像为文件（方便传到其他机器） ===
# 生成的 tar 文件约 15-20GB
export_image() {
    docker save alfworld-envtuning:latest | gzip > alfworld-envtuning.tar.gz
    echo "Image saved to alfworld-envtuning.tar.gz"
    echo "Size: $(du -sh alfworld-envtuning.tar.gz | awk '{print $1}')"
}

# === 第三步：在其他机器上加载镜像 ===
# scp alfworld-envtuning.tar.gz user@other-machine:/path/
# 然后在其他机器上：
load_image() {
    docker load < alfworld-envtuning.tar.gz
}

# === 第四步：运行实验 ===
# 模型通过 -v 挂载（不打包进镜像，节省空间）
# 如果其他机器没有模型，先用 huggingface-cli 下载

# 运行 vanilla GRPO
run_vanilla() {
    docker run --gpus all --ipc=host --shm-size=64g \
        -v /path/to/Qwen3-8B:/models/Qwen3-8B \
        -v /path/to/outputs:/workspace/outputs \
        alfworld-envtuning:latest \
        bash scripts/run_alfworld_vanilla.sh
}

# 运行 obs-aug GRPO
run_obs_aug() {
    docker run --gpus all --ipc=host --shm-size=64g \
        -v /path/to/Qwen3-8B:/models/Qwen3-8B \
        -v /path/to/outputs:/workspace/outputs \
        alfworld-envtuning:latest \
        bash scripts/run_alfworld_envtuning.sh
}

# 运行 full envtuning GRPO
run_full_envtuning() {
    docker run --gpus all --ipc=host --shm-size=64g \
        -v /path/to/Qwen3-8B:/models/Qwen3-8B \
        -v /path/to/outputs:/workspace/outputs \
        alfworld-envtuning:latest \
        bash scripts/run_alfworld_full_envtuning.sh
}

# === 用法 ===
echo "Usage:"
echo "  bash build_and_run.sh build     # 构建镜像"
echo "  bash build_and_run.sh export    # 导出镜像文件"
echo "  bash build_and_run.sh load      # 加载镜像（在其他机器上）"
echo ""
echo "运行实验（需要修改模型和输出路径）："
echo "  bash build_and_run.sh vanilla"
echo "  bash build_and_run.sh obs-aug"
echo "  bash build_and_run.sh full-envtuning"

case "${1}" in
    build) build_image ;;
    export) export_image ;;
    load) load_image ;;
    vanilla) run_vanilla ;;
    obs-aug) run_obs_aug ;;
    full-envtuning) run_full_envtuning ;;
    *) echo "Available commands: build, export, load, vanilla, obs-aug, full-envtuning" ;;
esac
